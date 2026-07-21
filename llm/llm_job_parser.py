from __future__ import annotations

import json
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from core.job_parser import ParsedJobPosting, parse_job_posting
from llm.schema import JOB_POSTING_SCHEMA
from llm.prompts import JOB_PARSING_PROMPT
from llm.providers import get_llm_provider, LLMProviderUnavailable


class LLMJobParserError(RuntimeError):
    pass


def parse_job_posting_with_llm(
    text: str,
    model: str = "gpt-4o-mini",
    llm_provider: str = "openai",
    fallback_to_rules: bool = True,
) -> ParsedJobPosting:
    """Parse a job posting with OpenAI Structured Outputs.

    The OpenAI response is constrained by JOB_POSTING_SCHEMA, and we validate it
    again locally before converting it to ParsedJobPosting. If the API call or
    validation fails, the rule-based parser can be used as a fallback.
    """
    try:
        parsed_dict = request_structured_parse(text=text, model=model, llm_provider=llm_provider)
        validate_job_posting_payload(parsed_dict)
        return payload_to_parsed_job_posting(parsed_dict, raw_text_length=len(text))
    except Exception as exc:
        if fallback_to_rules:
            return parse_job_posting(text)
        raise LLMJobParserError(str(exc)) from exc


def request_structured_parse(text: str, model: str, llm_provider: str) -> dict[str, Any]:
    try:
        provider = get_llm_provider(llm_provider)
        return provider.generate_structured(
            model=model,
            schema_name="job_posting_parse",
            schema=JOB_POSTING_SCHEMA,
            system_prompt=JOB_PARSING_PROMPT.strip(),
            user_prompt=text,
        )
    except (LLMProviderUnavailable, json.JSONDecodeError) as exc:
        raise LLMJobParserError(str(exc)) from exc


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
