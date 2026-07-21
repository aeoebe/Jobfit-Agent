from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from llm.providers.base import LLMProvider, LLMProviderUnavailable


class OpenAIProvider(LLMProvider):
    name = "openai"

    def is_available(self) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        return bool(api_key and api_key != "dummy-openai-api-key")

    def generate_structured(
        self,
        model: str,
        schema_name: str,
        schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        if not self.is_available():
            raise LLMProviderUnavailable("OPENAI_API_KEY is not configured.")

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMProviderUnavailable("OpenAI returned an empty response.")
        return json.loads(content)

    def embed(self, text: str, model: str) -> list[float]:
        if not self.is_available():
            raise LLMProviderUnavailable("OPENAI_API_KEY is not configured.")

        client = OpenAI()
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

