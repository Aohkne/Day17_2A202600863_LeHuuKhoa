from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    RECALL_REQUEST_RE,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    extract_profile_updates_with_confidence,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


# Bonus: only persist facts whose extraction pattern is reliable enough (see
# `memory_store.FACT_CONFIDENCE`). Raise this if a noisier dataset starts
# polluting `User.md`; lower it to recover recall at the cost of more false saves.
PROFILE_CONFIDENCE_THRESHOLD = 0.5

_FACT_LABELS = {
    "name": "tên là {}",
    "location": "hiện đang ở {}",
    "profession": "đang làm {}",
    "interest": "quan tâm tới {}",
    "drink": "đồ uống yêu thích là {}",
    "food": "món ăn yêu thích là {}",
    "pet": "đang nuôi {}",
    "style": "muốn mình trả lời theo phong cách {}",
    "style_format": "muốn định dạng câu trả lời theo {}",
}


class AdvancedAgent:
    """Agent B: within-session memory + persistent `User.md` + compact memory.

    Required memory layers:
    1. within-session memory (handled by the live agent's checkpointer, or by
       `CompactMemoryManager` in offline mode)
    2. persistent `User.md` via `UserProfileStore`
    3. compact memory for long threads via `CompactMemoryManager`
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._current_user_id: str | None = None

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            self._current_user_id = user_id
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]}, config=config
            )
            ai_message = result["messages"][-1]
            reply_text = ai_message.content
            usage = getattr(ai_message, "usage_metadata", None) or {}

            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + (
                usage.get("output_tokens") or estimate_tokens(reply_text)
            )
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + (
                usage.get("input_tokens") or estimate_tokens(message)
            )
            # Keep compact memory's bookkeeping (compaction count) consistent even in live mode.
            self.compact_memory.append(thread_id, "user", message)
            self.compact_memory.append(thread_id, "assistant", reply_text)

            return {
                "response": reply_text,
                "token_usage": self.thread_tokens[thread_id],
                "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
            }

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates_with_confidence(message, min_confidence=PROFILE_CONFIDENCE_THRESHOLD)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        reply_text = self._offline_response(user_id, thread_id, message)

        self.compact_memory.append(thread_id, "assistant", reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(reply_text)

        return {
            "response": reply_text,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        messages_tokens = sum(estimate_tokens(m["content"]) for m in ctx["messages"])
        return estimate_tokens(profile_text) + estimate_tokens(ctx["summary"]) + messages_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        text = message.strip()
        if not RECALL_REQUEST_RE.search(text):
            updates = extract_profile_updates(text)
            if updates:
                return "Mình đã lưu lại thông tin này vào hồ sơ của bạn."
            return "Mình đã ghi nhận, cảm ơn bạn đã chia sẻ."

        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin nào về bạn trong hồ sơ persistent."

        parts = [_FACT_LABELS[key].format(value) for key, value in facts.items() if key in _FACT_LABELS]
        answer = "Mình nhớ: " + "; ".join(parts) + "."

        summary = self.compact_memory.context(thread_id).get("summary", "")
        if summary:
            answer += f" {summary}"
        return answer

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        try:
            from langchain.agents import create_agent
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
            agent_self = self

            @tool
            def read_user_profile() -> str:
                """Đọc User.md (persistent profile) của người dùng hiện tại."""
                return agent_self.profile_store.read_text(agent_self._current_user_id)

            @tool
            def save_user_fact(key: str, value: str) -> str:
                """Lưu hoặc cập nhật một fact bền vững (key, value) vào User.md của người dùng hiện tại."""
                agent_self.profile_store.upsert_fact(agent_self._current_user_id, key, value)
                return f"saved {key}={value}"

            system_prompt = (
                "Bạn là trợ lý ghi nhớ thông tin người dùng lâu dài qua nhiều phiên. "
                "Trước khi trả lời câu hỏi về thông tin cá nhân, hãy gọi tool read_user_profile để đọc hồ sơ. "
                "Khi người dùng cung cấp fact ổn định (tên, nơi ở, nghề nghiệp, sở thích, "
                "style trả lời mong muốn), hãy gọi tool save_user_fact để lưu lại, dùng key "
                "tiếng Anh ngắn (name, location, profession, style, interest, drink, food, pet)."
            )

            return create_agent(
                model=model,
                tools=[read_user_profile, save_user_fact],
                system_prompt=system_prompt,
                checkpointer=InMemorySaver(),
            )
        except Exception:
            return None
