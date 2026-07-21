from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import uuid

from core.profile_loader import ProfileDocument, split_profile_documents
from llm.providers import get_llm_provider, LLMProviderUnavailable


VECTOR_DIMENSIONS = 128


class VectorStoreUnavailable(RuntimeError):
    pass


@dataclass
class VectorSearchResult:
    content: str
    source: str
    score: float


class ProfileVectorIndex:
    def __init__(self, profile_documents: list[ProfileDocument], embedding_provider: str = "local") -> None:
        self.chunks = split_profile_documents(profile_documents)
        self.embedding_provider = embedding_provider
        self.collection = create_chroma_collection()
        self._index_chunks()

    def query(self, text: str, top_k: int = 1) -> list[VectorSearchResult]:
        if not self.chunks:
            return []

        result = self.collection.query(
            query_embeddings=[embed_text(text, provider=self.embedding_provider)],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            search_results.append(
                VectorSearchResult(
                    content=document,
                    source=metadata.get("source", "unknown"),
                    score=distance_to_score(distance),
                )
            )

        return search_results

    def _index_chunks(self) -> None:
        if not self.chunks:
            return

        ids = [f"chunk-{index}" for index in range(len(self.chunks))]
        documents = [chunk.content for chunk in self.chunks]
        metadatas = [{"source": chunk.source} for chunk in self.chunks]
        embeddings = [embed_text(chunk.content, provider=self.embedding_provider) for chunk in self.chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def build_profile_vector_index(
    profile_documents: list[ProfileDocument],
    embedding_provider: str = "local",
) -> ProfileVectorIndex:
    return ProfileVectorIndex(profile_documents, embedding_provider=embedding_provider)


def create_chroma_collection():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise VectorStoreUnavailable("chromadb is not installed.") from exc

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    return client.create_collection(name=f"profile_{uuid.uuid4().hex}")


def embed_text(text: str, provider: str = "local") -> list[float]:
    if provider == "openai":
        return embed_text_with_openai(text)
    if provider == "ollama":
        return embed_text_with_ollama(text)
    return embed_text_locally(text)


def embed_text_with_openai(text: str) -> list[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy-openai-api-key":
        raise VectorStoreUnavailable("OPENAI_API_KEY is not set for OpenAI embeddings.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise VectorStoreUnavailable("openai package is not installed.") from exc

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def embed_text_with_ollama(text: str) -> list[float]:
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    try:
        return get_llm_provider("ollama").embed(text=text, model=model)
    except LLMProviderUnavailable as exc:
        raise VectorStoreUnavailable(str(exc)) from exc


def embed_text_locally(text: str) -> list[float]:
    """Create a deterministic local embedding for MVP retrieval.

    This avoids external embedding APIs while still exercising a real vector DB.
    """
    vector = [0.0] * VECTOR_DIMENSIONS

    for token in tokenize_for_embedding(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % VECTOR_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def tokenize_for_embedding(text: str) -> list[str]:
    import re

    return [
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
        if len(token) >= 2
    ]


def distance_to_score(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def vector_score_to_match_score(score: float) -> int:
    if score >= 0.8:
        return 5
    if score >= 0.65:
        return 4
    if score >= 0.5:
        return 3
    if score >= 0.35:
        return 2
    return 0
