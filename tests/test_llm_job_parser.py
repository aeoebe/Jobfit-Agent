import pytest

from llm.llm_job_parser import (
    LLMJobParserError,
    payload_to_parsed_job_posting,
    validate_job_posting_payload,
)


def test_validate_job_posting_payload_accepts_valid_payload():
    payload = {
        "role": "AI Agent Engineer",
        "required_skills": ["Python", "RAG"],
        "preferred_skills": ["LangGraph"],
        "responsibilities": ["Build agent workflows."],
        "seniority": "mid",
        "keywords": ["agent", "evaluation"],
    }

    validate_job_posting_payload(payload)
    parsed = payload_to_parsed_job_posting(payload, raw_text_length=100)

    assert parsed.role == "AI Agent Engineer"
    assert parsed.required_skills == ["Python", "RAG"]
    assert parsed.raw_text_length == 100


def test_validate_job_posting_payload_rejects_invalid_payload():
    payload = {
        "role": "AI Agent Engineer",
        "required_skills": "Python, RAG",
        "preferred_skills": [],
        "responsibilities": [],
        "seniority": "mid",
        "keywords": [],
    }

    with pytest.raises(LLMJobParserError):
        validate_job_posting_payload(payload)

