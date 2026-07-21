import pytest

from llm.providers import LLMProviderUnavailable, get_llm_provider


def test_get_llm_provider_returns_openai_provider():
    provider = get_llm_provider("openai")

    assert provider.name == "openai"


def test_get_llm_provider_returns_ollama_provider():
    provider = get_llm_provider("ollama")

    assert provider.name == "ollama"


def test_get_llm_provider_rejects_unknown_provider():
    with pytest.raises(LLMProviderUnavailable):
        get_llm_provider("unknown")
