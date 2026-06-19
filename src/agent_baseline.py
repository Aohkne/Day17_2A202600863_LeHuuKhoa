from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _sentence_overlap_answer(prior_messages: list[dict[str, str]], question: str) -> str:
    """Naive within-session retrieval: pick the prior user sentence with the most word overlap.

    This is the only "memory" baseline has - it never looks outside the current
    thread's own message list, so a fresh thread always falls back to the default.
    """

    question_words = set(_WORD_RE.findall(question.lower()))
    best_sentence, best_overlap = "", 0
    for m in prior_messages:
        if m["role"] != "user":
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", m["content"]):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(question_words & set(_WORD_RE.findall(sentence.lower())))
            if overlap > best_overlap:
                best_overlap, best_sentence = overlap, sentence

    if best_overlap >= 2:
        return best_sentence
    return "Mình chưa có đủ thông tin trong phiên này để trả lời câu này."


class BaselineAgent:
    """Agent A: within-session memory only, no persistent `User.md`.

    A new thread id is a blank slate even for the same user, which is the
    fair baseline the advanced agent (with persistent memory) is compared against.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]}, config=config
            )
            ai_message = result["messages"][-1]
            reply_text = ai_message.content
            usage = getattr(ai_message, "usage_metadata", None) or {}

            session = self.sessions.setdefault(thread_id, SessionState())
            session.token_usage += usage.get("output_tokens") or estimate_tokens(reply_text)
            session.prompt_tokens_processed += usage.get("input_tokens") or estimate_tokens(message)

            return {
                "response": reply_text,
                "token_usage": session.token_usage,
                "prompt_tokens_processed": session.prompt_tokens_processed,
            }

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        prior_messages = list(session.messages)
        session.messages.append({"role": "user", "content": message})

        text = message.strip()
        if text.endswith("?"):
            reply_text = _sentence_overlap_answer(prior_messages, text)
        else:
            reply_text = "Mình đã ghi nhận thông tin bạn vừa chia sẻ trong phiên này."

        session.messages.append({"role": "assistant", "content": reply_text})

        # No compaction here: baseline re-processes its entire growing history every turn.
        session.prompt_tokens_processed += sum(estimate_tokens(m["content"]) for m in session.messages)
        session.token_usage += estimate_tokens(reply_text)

        return {
            "response": reply_text,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
            return create_agent(model=model, checkpointer=InMemorySaver())
        except Exception:
            return None
