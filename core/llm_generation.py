from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI

from core.job_parser import ParsedJobPosting
from core.matcher import RequirementMatch


class LLMGenerationUnavailable(RuntimeError):
    pass


LEARNING_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "actions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["actions"],
}

RESUME_SUGGESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement": {"type": "string"},
                    "suggested_bullet": {"type": "string"},
                },
                "required": ["requirement", "suggested_bullet"],
            },
        }
    },
    "required": ["suggestions"],
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "markdown": {"type": "string"},
    },
    "required": ["markdown"],
}


def has_real_openai_api_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    return bool(api_key and api_key != "dummy-openai-api-key")


def generate_learning_plan_with_llm(
    parsed_job: ParsedJobPosting,
    gaps: list[str],
    model: str,
) -> tuple[list[str], bool]:
    if not has_real_openai_api_key():
        return [], False

    prompt = f"""
Create a concise learning plan for a candidate applying to this role.

Role: {parsed_job.role}
Seniority: {parsed_job.seniority}
Gaps: {gaps}

Return 3-5 practical actions. Each action should be specific and portfolio-oriented.
""".strip()

    try:
        payload = request_structured_generation(
            model=model,
            schema_name="learning_plan",
            schema=LEARNING_PLAN_SCHEMA,
            system_prompt="You create practical, grounded job preparation plans.",
            user_prompt=prompt,
        )
        return payload["actions"], True
    except LLMGenerationUnavailable:
        return [], False


def generate_resume_suggestions_with_llm(
    parsed_job: ParsedJobPosting,
    matches: list[RequirementMatch],
    model: str,
) -> tuple[list[dict[str, str]], bool]:
    if not has_real_openai_api_key():
        return [], False

    usable_matches = [
        {
            "requirement": match.requirement,
            "source": match.source,
            "evidence": match.matched_evidence,
            "score": match.score,
        }
        for match in matches
        if not match.gap
    ][:6]

    prompt = f"""
Suggest resume bullets for this job application using only the provided evidence.

Role: {parsed_job.role}
Responsibilities: {parsed_job.responsibilities}
Evidence matches: {usable_matches}

Rules:
- Do not invent metrics, tools, employers, or outcomes.
- Each bullet must be grounded in one evidence item.
- Keep bullets concise and action-oriented.
""".strip()

    try:
        payload = request_structured_generation(
            model=model,
            schema_name="resume_suggestions",
            schema=RESUME_SUGGESTIONS_SCHEMA,
            system_prompt="You rewrite resume bullets using only grounded evidence.",
            user_prompt=prompt,
        )
        return payload["suggestions"], True
    except LLMGenerationUnavailable:
        return [], False


def generate_report_with_llm(
    template_report: str,
    model: str,
) -> tuple[str | None, bool]:
    if not has_real_openai_api_key():
        return None, False

    prompt = f"""
Refine this markdown job-fit report for clarity and usefulness.

Rules:
- Preserve the same factual content.
- Do not invent new experience, scores, or evidence.
- Keep markdown headings.
- Keep it concise.

Report:
{template_report}
""".strip()

    try:
        payload = request_structured_generation(
            model=model,
            schema_name="jobfit_report",
            schema=REPORT_SCHEMA,
            system_prompt="You improve markdown reports without adding unsupported facts.",
            user_prompt=prompt,
        )
        return payload["markdown"], True
    except LLMGenerationUnavailable:
        return None, False


def request_structured_generation(
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    try:
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
            raise LLMGenerationUnavailable("Model returned an empty response.")

        payload = json.loads(content)
        validate(instance=payload, schema=schema)
        return payload
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMGenerationUnavailable(str(exc)) from exc
    except Exception as exc:
        raise LLMGenerationUnavailable(str(exc)) from exc

