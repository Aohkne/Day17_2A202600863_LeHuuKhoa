from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return len(text.strip())


DEFAULT_PROFILE_TEMPLATE = "# User Profile\n"

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_FACT_LINE_RE = re.compile(r"^- (\w+): (.*)$", re.MULTILINE)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one markdown file per user id."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = _SLUG_RE.sub("_", user_id.strip().lower()).strip("_") or "user"
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE_TEMPLATE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(_FACT_LINE_RE.findall(self.read_text(user_id)))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        content = self.read_text(user_id)
        new_line = f"- {key}: {value}"
        line_pattern = re.compile(rf"^- {re.escape(key)}: .*$", re.MULTILINE)
        if line_pattern.search(content):
            updated = line_pattern.sub(new_line, content, count=1)
        else:
            updated = content.rstrip("\n") + "\n" + new_line + "\n"
        return self.write_text(user_id, updated)


_NAME_STOP = r"(?:và|nhưng|mà|chứ|nữa|để|cho|vì|dù|trong)\b|[,.\!?\n]|$"
_NAME_RE = re.compile(rf"tên (?:là|mình là)\s+([^.,!?\n]+?)(?=\s+{_NAME_STOP})", re.IGNORECASE)

_LOCATION_STOP = r"(?:và|nhưng|mà|chứ|nữa|để|cho|vì|dù|trong)\b|[,.\!\n]|$"
_LOCATION_POS_RE = re.compile(rf"\bở\s+([^.,!\n]+?)(?=\s+{_LOCATION_STOP})", re.IGNORECASE)
_LOCATION_NEG_RE = re.compile(rf"không còn ở\s+([^.,!\n]+?)(?=\s+{_LOCATION_STOP})", re.IGNORECASE)

_PROFESSION_STOP = r"(?:cho|ở|nữa|vì|chứ|và|nhưng|mà|dù|trong)\b|[,.\!\n]|$"
_PROFESSION_POS_RE = re.compile(
    rf"(?:chuyển sang|làm|nghề(?: nghiệp)?[^.\n]*?là)\s+([^.,!\n]+?)(?=\s+{_PROFESSION_STOP})",
    re.IGNORECASE,
)
_PROFESSION_NEG_RE = re.compile(
    rf"không còn (?:làm|là)\s+([^.,!\n]+?)(?=\s+{_PROFESSION_STOP})", re.IGNORECASE
)
_PROFESSION_SUFFIXES = ("engineer", "manager", "developer", "scientist", "designer", "analyst")

_DRINK_RE = re.compile(r"đồ uống yêu thích là\s+([^.,!\n]+)", re.IGNORECASE)
_FOOD_RE = re.compile(r"món ăn yêu thích là\s+([^.,!\n]+)", re.IGNORECASE)
_PET_RE = re.compile(r"nuôi (?:một |)(?:bé |con |)([a-zà-ỹ]+)", re.IGNORECASE)

_STYLE_KEYWORDS = ("ngắn gọn", "súc tích")
_FORMAT_KEYWORDS = ("3 bullet", "bullet")

_INTEREST_TRIGGER_RE = re.compile(r"thích|quan tâm", re.IGNORECASE)
_INTEREST_KEYWORDS = ("Python", "AI")  # case-sensitive: lowercase "ai" means "who" in Vietnamese

# A "recall request" isn't always a literal "?" question - imperatives like
# "Nhắc lại..." / "Tóm tắt..." ask the agent to recall, not provide, info.
# Skipping these prevents nonsense like "...mình nuôi con gì." being parsed
# as a new pet fact ("gì") that overwrites the real one.
RECALL_REQUEST_RE = re.compile(
    r"\?|nhắc lại|nhắc giúp|hãy nhắc|tóm tắt|nói lại|cho (?:mình|tôi) biết",
    re.IGNORECASE,
)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert one raw user message into stable profile facts.

    Heuristic, regex-based extraction tuned for the lab's Vietnamese dataset.
    Recall-request turns are skipped so the agent never overwrites a fact just
    because the user asked the agent to recall it.
    """

    text = message.strip()
    if not text or RECALL_REQUEST_RE.search(text):
        return {}

    facts: dict[str, str] = {}
    lowered = text.lower()

    name_match = _NAME_RE.search(text)
    if name_match:
        facts["name"] = name_match.group(1).strip()

    negated_locations = {m.strip() for m in _LOCATION_NEG_RE.findall(text)}
    location_candidates = [
        m.strip()
        for m in _LOCATION_POS_RE.findall(text)
        if m.strip() and m.strip()[0].isupper() and m.strip() not in negated_locations
    ]
    if location_candidates:
        facts["location"] = location_candidates[-1]

    negated_professions = {m.strip() for m in _PROFESSION_NEG_RE.findall(text)}
    profession_candidates = []
    for m in _PROFESSION_POS_RE.findall(text):
        candidate = m.strip()
        if not candidate or candidate in negated_professions:
            continue
        if candidate.lower().split()[-1] in _PROFESSION_SUFFIXES:
            profession_candidates.append(candidate)
    if profession_candidates:
        facts["profession"] = profession_candidates[-1]

    drink_match = _DRINK_RE.search(text)
    if drink_match:
        facts["drink"] = drink_match.group(1).strip()

    food_match = _FOOD_RE.search(text)
    if food_match:
        facts["food"] = food_match.group(1).strip()

    pet_match = _PET_RE.search(text)
    if pet_match:
        facts["pet"] = pet_match.group(1).strip()

    if any(keyword in lowered for keyword in _STYLE_KEYWORDS):
        facts["style"] = "ngắn gọn"

    if any(keyword in lowered for keyword in _FORMAT_KEYWORDS):
        facts["style_format"] = "3 bullet" if "3 bullet" in lowered else "bullet"

    if _INTEREST_TRIGGER_RE.search(text):
        found_interests = [kw for kw in _INTEREST_KEYWORDS if re.search(rf"\b{kw}\b", text)]
        if found_interests:
            facts["interest"] = ", ".join(found_interests)

    return facts


# Bonus: confidence threshold. Each extraction pattern gets a reliability score -
# anchored, hard-to-misfire patterns (drink/food/name "X là Y") score high; loosely
# triggered heuristics (pet, interest) score lower since they're more likely to
# catch noise (e.g. "nuôi con gì?"). Callers that want a safety margin against
# polluting `User.md` can raise `min_confidence` instead of trusting every match.
FACT_CONFIDENCE = {
    "name": 0.95,
    "drink": 0.95,
    "food": 0.95,
    "style": 0.9,
    "style_format": 0.85,
    "location": 0.85,
    "profession": 0.8,
    "interest": 0.7,
    "pet": 0.6,
}


def extract_profile_updates_with_confidence(message: str, min_confidence: float = 0.0) -> dict[str, str]:
    """Same extraction as `extract_profile_updates`, dropping facts below `min_confidence`."""

    facts = extract_profile_updates(message)
    return {key: value for key, value in facts.items() if FACT_CONFIDENCE.get(key, 0.0) >= min_confidence}


# Bao nhêu thì sumarize và lấy bao nhiêu tin nhắn gần nhất
def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compact summary: first sentence of each of the last `max_items` messages."""

    if not messages:
        return ""

    chunks = []
    for m in messages[-max_items:]:
        snippet = m["content"].strip().split(".")[0].strip()
        if snippet:
            chunks.append(f"{m['role']}: {snippet}")

    return "Tóm tắt trước đó: " + " | ".join(chunks)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the most recent `keep_messages` in full and folds older content
    into a running heuristic summary once the thread's estimated token count
    exceeds `threshold_tokens`.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread_state(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._thread_state(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]
        messages.append({"role": role, "content": content})

        total_tokens = estimate_tokens(thread_state["summary"]) + sum(
            estimate_tokens(m["content"]) for m in messages
        )
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            old_messages = messages[: -self.keep_messages]
            new_summary_piece = summarize_messages(old_messages)
            thread_state["summary"] = f"{thread_state['summary']} {new_summary_piece}".strip()
            thread_state["messages"] = messages[-self.keep_messages :]
            thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread_state(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
