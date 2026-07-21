from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProviderUnavailable(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        model: str,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

