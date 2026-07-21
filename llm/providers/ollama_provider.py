from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from llm.providers.base import LLMProvider, LLMProviderUnavailable


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def is_available(self) -> bool:
        try:
            self._get("/api/version")
            return True
        except LLMProviderUnavailable:
            return False

    def generate_structured(
        self,
        model: str,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        prompt = f"{system_prompt}\n\n{user_prompt}\n\nRespond with JSON that matches the schema."
        payload = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = self._post("/api/generate", payload)
        content = response.get("response", "")
        if not content:
            raise LLMProviderUnavailable("Ollama returned an empty response.")
        return json.loads(content)

    def embed(self, text: str, model: str) -> list[float]:
        try:
            response = self._post("/api/embed", {"model": model, "input": text})
            embeddings = response.get("embeddings")
            if embeddings:
                return embeddings[0]
        except LLMProviderUnavailable:
            pass

        response = self._post("/api/embeddings", {"model": model, "prompt": text})
        embedding = response.get("embedding")
        if not embedding:
            raise LLMProviderUnavailable("Ollama returned no embedding.")
        return embedding

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise LLMProviderUnavailable(f"Ollama request failed: {exc}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise LLMProviderUnavailable(f"Ollama request failed: {exc}") from exc
