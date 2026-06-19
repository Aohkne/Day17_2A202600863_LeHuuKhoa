from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for fact in expected if fact in answer)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    text = answer.strip()
    if not text:
        return 0.0

    score = 0.4  # produced a non-empty answer at all
    if any(fact in text for fact in expected):
        score += 0.4

    word_count = len(text.split())
    if 3 <= word_count <= 60:
        score += 0.2

    return min(score, 1.0)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed every conversation to `agent`, then test cross-session recall in a fresh thread per conversation."""

    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    has_memory_file = hasattr(agent, "memory_file_size")
    memory_before: dict[str, int] = {}
    memory_after: dict[str, int] = {}

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        if has_memory_file and user_id not in memory_before:
            memory_before[user_id] = agent.memory_file_size(user_id)

        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)

        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)

        recall_thread_id = f"{thread_id}-recall"
        for question in conv.get("recall_questions", []):
            result = agent.reply(user_id, recall_thread_id, question["question"])
            answer = result["response"]
            recall_scores.append(recall_points(answer, question["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, question["expected_contains"]))

        total_agent_tokens += agent.token_usage(recall_thread_id)
        total_prompt_tokens += agent.prompt_token_usage(recall_thread_id)
        total_compactions += agent.compaction_count(recall_thread_id)

        if has_memory_file:
            memory_after[user_id] = agent.memory_file_size(user_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    memory_growth = sum(memory_after.get(uid, 0) - memory_before.get(uid, 0) for uid in memory_before)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def _run_benchmark_section(title: str, conversations, config, state_subdir: str, force_offline: bool) -> None:
    config.state_dir = config.base_dir / "state" / state_subdir
    shutil.rmtree(config.state_dir, ignore_errors=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    baseline = BaselineAgent(config, force_offline=force_offline)
    advanced = AdvancedAgent(config, force_offline=force_offline)

    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config),
        run_agent_benchmark("Advanced", advanced, conversations, config),
    ]

    print(f"=== {title} ===")
    print(format_rows(rows))
    print()


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    # Real provider calls cost time/money across ~2 agents x many turns; default to the
    # deterministic offline path. Set BENCHMARK_LIVE=1 to exercise the real LLM provider.
    force_offline = os.environ.get("BENCHMARK_LIVE", "0") != "1"

    standard_conversations = load_conversations(config.data_dir / "conversations.json")
    stress_conversations = load_conversations(config.data_dir / "advanced_long_context.json")

    _run_benchmark_section("Standard Benchmark", standard_conversations, config, "benchmark_standard", force_offline)
    _run_benchmark_section(
        "Long-Context Stress Benchmark", stress_conversations, config, "benchmark_stress", force_offline
    )


if __name__ == "__main__":
    main()
