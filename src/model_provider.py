from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "nvidia": "custom",
    "nim": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map common aliases/typos (e.g. `anthorpic`, `custome`) onto a supported provider name."""

    key = (value or "").strip().lower()
    normalized = _PROVIDER_ALIASES.get(key, key)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider '{value}'. Supported: {list(SUPPORTED_PROVIDERS)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model for the selected provider.

    Each branch imports lazily so the lab can run in offline mode without
    every provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("Provider 'custom' requires base_url (e.g. LLM_BASE_URL).")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        # OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works directly.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported provider: {provider}")
