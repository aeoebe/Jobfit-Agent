from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI

from core.job_parser import ParsedJobPosting, parse_job_posting
from llm.schema import JOB_POSTING_SCHEMA
from llm.prompts import JOB_PARSING_PROMPT


class LLMJobParserError(RuntimeError):
    pass


def parse_job_posting_with_llm(
    text: str,
    model: str = "gpt-4o-mini",
    fallback_to_rules: bool = True,
) -> ParsedJobPosting:
    """Parse a job posting with OpenAI Structured Outputs.

    The OpenAI response is constrained by JOB_POSTING_SCHEMA, and we validate it
    again locally before converting it to ParsedJobPosting. If the API call or
    validation fails, the rule-based parser can be used as a fallback.
    """
    try:
        parsed_dict = request_structured_parse(text=text, model=model)
        validate_job_posting_payload(parsed_dict)
        return payload_to_parsed_job_posting(parsed_dict, raw_text_length=len(text))
    except Exception as exc:
        if fallback_to_rules:
            return parse_job_posting(text)
        raise LLMJobParserError(str(exc)) from exc


def request_structured_parse(text: str, model: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise LLMJobParserError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JOB_PARSING_PROMPT.strip()},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "job_posting_parse",
                "strict": True,
                "schema": JOB_POSTING_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise LLMJobParserError("Model returned an empty response.")

    return json.loads(content)


def validate_job_posting_payload(payload: dict[str, Any]) -> None:
    try:
        validate(instance=payload, schema=JOB_POSTING_SCHEMA)
    except ValidationError as exc:
        raise LLMJobParserError(f"Invalid structured output: {exc.message}") from exc


def payload_to_parsed_job_posting(payload: dict[str, Any], raw_text_length: int) -> ParsedJobPosting:
    return ParsedJobPosting(
        role=payload["role"],
        required_skills=payload["required_skills"],
        preferred_skills=payload["preferred_skills"],
        responsibilities=payload["responsibilities"],
        seniority=payload["seniority"],
        keywords=payload["keywords"],
        raw_text_length=raw_text_length,
    )

