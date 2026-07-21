from __future__ import annotations

from llm.providers.base import LLMProvider, LLMProviderUnavailable
from llm.providers.ollama_provider import OllamaProvider
from llm.providers.openai_provider import OpenAIProvider


def get_llm_provider(name: str = "openai") -> LLMProvider:
    normalized = (name or "openai").lower()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized == "ollama":
        return OllamaProvider()
    raise LLMProviderUnavailable(f"Unsupported LLM provider: {name}")


def has_available_llm_provider(name: str = "openai") -> bool:
    try:
        return get_llm_provider(name).is_available()
    except LLMProviderUnavailable:
        return False

