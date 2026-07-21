from __future__ import annotations

import json
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError

from core.job_parser import ParsedJobPosting
from core.matcher import RequirementMatch
from llm.prompts import (
    LEARNING_PLAN_PROMPT,
    RESUME_SUGGESTION_PROMPT,
    GENERATE_REPORT_PROMPT,
    GENERATE_COVER_LETTER_PROMPT,
    CRITIQUE_COVER_LETTER_PROMPT
)
from llm.schema import (
    LEARNING_PLAN_SCHEMA,
    REPORT_SCHEMA,
    RESUME_SUGGESTIONS_SCHEMA,
    COVER_LETTER_SCHEMA,
    CRITIC_SCHEMA
)
from llm.providers import get_llm_provider, has_available_llm_provider, LLMProviderUnavailable


class LLMGenerationUnavailable(RuntimeError):
    pass




def has_real_openai_api_key() -> bool:
    return has_available_llm_provider("openai")


def generate_learning_plan_with_llm(
    parsed_job: ParsedJobPosting,
    gaps: list[str],
    model: str,
    llm_provider: str = "openai",
) -> tuple[list[str], bool]:
    if not has_available_llm_provider(llm_provider):
        return [], False

    prompt = LEARNING_PLAN_PROMPT.format(
        role=parsed_job.role,
        seniority=parsed_job.seniority,
        gaps=gaps,
    ).strip()

    try:
        payload = request_structured_generation(
            model=model,
            llm_provider=llm_provider,
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
    llm_provider: str = "openai",
) -> tuple[list[dict[str, str]], bool]:
    if not has_available_llm_provider(llm_provider):
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

    prompt = RESUME_SUGGESTION_PROMPT.format(
        role=parsed_job.role,
        responsibilities = parsed_job.responsibilities,
        matches = usable_matches
    ).strip()

    try:
        payload = request_structured_generation(
            model=model,
            llm_provider=llm_provider,
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
    llm_provider: str = "openai",
) -> tuple[str | None, bool]:
    if not has_available_llm_provider(llm_provider):
        return None, False

    prompt = GENERATE_REPORT_PROMPT.format(
        template= template_report
    ).strip()

    try:
        payload = request_structured_generation(
            model=model,
            llm_provider=llm_provider,
            schema_name="jobfit_report",
            schema=REPORT_SCHEMA,
            system_prompt="You improve markdown reports without adding unsupported facts.",
            user_prompt=prompt,
        )
        return payload["markdown"], True
    except LLMGenerationUnavailable:
        return None, False
    
def generate_cover_letter_with_llm(
    parsed_job: ParsedJobPosting,
    matches: list[RequirementMatch],
    gaps: list[str],
    model: str,
    critic_feedback: str = "",
    llm_provider: str = "openai",
) -> tuple[str, bool]:
    if not has_available_llm_provider(llm_provider):
        return "", False
 
    feedback_section = f"\n\nPrevious critic feedback to address:\n{critic_feedback}" if critic_feedback else ""

    evidence = [
    {
        "requirement": m.requirement,
        "evidence": m.matched_evidence,
        "source": m.source,
        "score": m.score,
    }
    for m in matches
    if not m.gap
][:10]
 
    prompt = GENERATE_COVER_LETTER_PROMPT.format(
        role=parsed_job.role,
        seniority=parsed_job.seniority,
        evidence=json.dumps(evidence, indent=2, ensure_ascii=False,),
        gaps="\n".join(gaps),
        feedback=feedback_section
    ).strip()
 
    try:
        payload = request_structured_generation(
            model=model,
            llm_provider=llm_provider,
            schema_name="cover_letter",
            schema=COVER_LETTER_SCHEMA,
            system_prompt="You write compelling, honest cover letters grounded in real evidence.",
            user_prompt=prompt,
        )
        return payload["cover_letter"], True
    except LLMGenerationUnavailable:
        return "", False
 
 
def critique_cover_letter_with_llm(
    parsed_job: ParsedJobPosting,
    cover_letter: str,
    model: str,
    llm_provider: str = "openai",
) -> tuple[int, str, bool]:
    if not has_available_llm_provider(llm_provider):
        return 0, "", False
 
    prompt = CRITIQUE_COVER_LETTER_PROMPT.format(
        required_skills=parsed_job.required_skills,
        preferred_skills=parsed_job.preferred_skills,
        responsibilities=parsed_job.responsibilities,
        cv=cover_letter
    ).strip()
 
    try:
        payload = request_structured_generation(
            model=model,
            llm_provider=llm_provider,
            schema_name="critic",
            schema=CRITIC_SCHEMA,
            system_prompt="You are a strict recruiter evaluating cover letters against job requirements.",
            user_prompt=prompt,
        )
        score = payload["score"]

        if not 0 <= score <= 100:
            raise LLMGenerationUnavailable(
                f"Invalid critic score: {score}"
            )

        return score, payload["feedback"], True
    except LLMGenerationUnavailable:
        return 0, "", False


def request_structured_generation(
    model: str,
    llm_provider: str,
    schema_name: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    try:
        provider = get_llm_provider(llm_provider)
        payload = provider.generate_structured(
            model=model,
            schema_name=schema_name,
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        validate(instance=payload, schema=schema)
        return payload
    except (ValidationError, json.JSONDecodeError, LLMProviderUnavailable) as exc:
        raise LLMGenerationUnavailable(str(exc)) from exc
    except Exception as exc:
        raise LLMGenerationUnavailable(str(exc)) from exc
