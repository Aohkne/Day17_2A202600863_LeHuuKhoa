from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    dummy_model = ProviderConfig(provider="custom", model_name="test-model", temperature=0.0)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "user1"

    # No file yet -> default template, not a crash.
    default_text = agent.profile_store.read_text(user_id)
    assert "User Profile" in default_text
    assert agent.profile_store.file_size(user_id) == 0

    agent.profile_store.write_text(user_id, "# User Profile\n- name: Khoa\n")
    assert "Khoa" in agent.profile_store.read_text(user_id)
    assert agent.profile_store.file_size(user_id) > 0

    changed = agent.profile_store.edit_text(user_id, "Khoa", "KhoaLe")
    assert changed is True
    assert "KhoaLe" in agent.profile_store.read_text(user_id)
    assert agent.profile_store.edit_text(user_id, "NotPresent", "X") is False

    # upsert_fact is idempotent: a correction replaces the old value in place.
    agent.profile_store.upsert_fact(user_id, "location", "Hanoi")
    agent.profile_store.upsert_fact(user_id, "location", "Da Nang")
    content = agent.profile_store.read_text(user_id)
    assert content.count("- location:") == 1
    assert "Da Nang" in content
    assert agent.profile_store.facts(user_id)["location"] == "Da Nang"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id, thread_id = "user1", "stress-thread"

    long_message = "Mình muốn chia sẻ một đoạn nội dung khá dài để chắc chắn vượt threshold token nén lại. " * 5
    for _ in range(10):
        agent.reply(user_id, thread_id, long_message)

    assert agent.compaction_count(thread_id) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "user1"

    baseline.reply(user_id, "thread-1", "Chào bạn, mình tên là Khoa.")
    advanced.reply(user_id, "thread-1", "Chào bạn, mình tên là Khoa.")

    baseline_answer = baseline.reply(user_id, "thread-2", "Nhắc lại tên mình giúp.")["response"]
    advanced_answer = advanced.reply(user_id, "thread-2", "Nhắc lại tên mình giúp.")["response"]

    assert "Khoa" not in baseline_answer
    assert "Khoa" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id, thread_id = "user1", "long-thread"

    long_message = "Đây là một đoạn nội dung khá dài được lặp lại nhiều lần để mô phỏng hội thoại dài. " * 4
    for _ in range(15):
        baseline.reply(user_id, thread_id, long_message)
        advanced.reply(user_id, thread_id, long_message)

    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
