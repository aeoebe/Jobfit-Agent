from core.job_parser import parse_job_posting


def test_parse_job_posting_extracts_core_fields():
    text = """
AI Agent Engineer

Responsibilities:
- Build LLM agent workflows.
- Create evaluation datasets.

Required Qualifications:
- Strong Python experience.
- Experience with OpenAI APIs and RAG.

Preferred Qualifications:
- Experience with LangGraph and Chroma.
"""

    parsed = parse_job_posting(text)

    assert parsed.role == "AI Agent Engineer"
    assert "Python" in parsed.required_skills
    assert "OpenAI" in parsed.required_skills
    assert "RAG" in parsed.required_skills
    assert "LangGraph" in parsed.preferred_skills
    assert "Chroma" in parsed.preferred_skills
    assert parsed.responsibilities


def test_parse_job_posting_extracts_unlisted_skills_and_role():
    text = """
Search Platform Engineer

What you will do:
- Build large-scale search and recommendation services.

Requirements:
- Strong Rust experience.
- Experience with Temporal, Weaviate, and Ray Serve.

Nice to have:
- Familiarity with Vespa or OpenTelemetry.
"""

    parsed = parse_job_posting(text)

    assert parsed.role == "Search Platform Engineer"
    assert "Rust" in parsed.required_skills
    assert "Temporal" in parsed.required_skills
    assert "Weaviate" in parsed.required_skills
    assert "Ray Serve" in parsed.required_skills
    assert "Vespa" in parsed.preferred_skills
    assert "OpenTelemetry" in parsed.preferred_skills
