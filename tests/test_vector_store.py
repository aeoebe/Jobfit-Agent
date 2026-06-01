from core.profile_loader import ProfileDocument
import pytest

from core.vector_store import (
    VectorStoreUnavailable,
    build_profile_vector_index,
    embed_text,
    embed_text_with_openai,
)


def test_embed_text_is_deterministic_and_normalized():
    first = embed_text("Python RAG LangGraph")
    second = embed_text("Python RAG LangGraph")

    assert first == second
    assert len(first) == 128
    assert any(value != 0 for value in first)


def test_profile_vector_index_returns_profile_evidence():
    documents = [
        ProfileDocument(
            source="projects.md",
            content="## Retrieval Project\n\nBuilt a RAG document QA system with chunk retrieval.",
        )
    ]

    index = build_profile_vector_index(documents)
    results = index.query("RAG", top_k=1)

    assert results
    assert results[0].source == "projects.md"
    assert "RAG" in results[0].content


def test_openai_embedding_requires_real_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-api-key")

    with pytest.raises(VectorStoreUnavailable):
        embed_text_with_openai("RAG")
