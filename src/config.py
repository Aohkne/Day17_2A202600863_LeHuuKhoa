from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is an optional convenience
    load_dotenv = None


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths for the repo root, dataset directory, and state directory.
    - Compact-memory settings (threshold and number of messages to keep).
    - Provider settings for the main model and the judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `{prefix}PROVIDER` / `{prefix}LLM_*` env vars into a ProviderConfig.

    Generic knobs (PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL) work for any
    of the supported providers since `api_key`/`base_url` are optional per provider.
    `fallback` lets the judge model reuse the main model's settings when its own
    `JUDGE_*` vars are not set.
    """

    provider_raw = os.getenv(f"{prefix}PROVIDER") or (fallback.provider if fallback else "openai")
    model_name = os.getenv(f"{prefix}LLM_MODEL") or (fallback.model_name if fallback else "gpt-4o-mini")
    api_key = os.getenv(f"{prefix}LLM_API_KEY") or (fallback.api_key if fallback else None)
    base_url = os.getenv(f"{prefix}LLM_BASE_URL") or (fallback.base_url if fallback else None)

    temperature_raw = os.getenv(f"{prefix}LLM_TEMPERATURE")
    if temperature_raw is not None:
        temperature = float(temperature_raw)
    else:
        temperature = fallback.temperature if fallback else 0.2

    return ProviderConfig(
        provider=normalize_provider(provider_raw),
        model_name=model_name,
        temperature=temperature,
        api_key=api_key or None,
        base_url=base_url or None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    if load_dotenv is not None:
        load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config_from_env("")
    judge_model = _provider_config_from_env("JUDGE_", fallback=model)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
