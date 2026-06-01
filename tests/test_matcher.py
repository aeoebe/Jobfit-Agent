from core.job_parser import ParsedJobPosting
from core.matcher import match_job_to_profile, summarize_matches
from core.profile_loader import ProfileDocument


def test_match_job_to_profile_finds_evidence_and_gaps():
    parsed_job = ParsedJobPosting(
        role="AI Agent Engineer",
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG"],
        responsibilities=[],
        seniority="unknown",
        keywords=[],
        raw_text_length=100,
    )
    documents = [
        ProfileDocument(
            source="resume.md",
            content="## Skills\n\nPython\n\n## Project\n\nBuilt a RAG document QA system.",
        )
    ]

    matches = match_job_to_profile(parsed_job, documents)
    summary = summarize_matches(matches)

    python_match = next(match for match in matches if match.requirement == "Python")
    langgraph_match = next(match for match in matches if match.requirement == "LangGraph")

    assert python_match.score == 5
    assert not python_match.gap
    assert langgraph_match.gap
    assert summary["total_count"] == 3
    assert summary["gap_count"] == 1


def test_match_job_to_profile_can_use_vector_retrieval():
    parsed_job = ParsedJobPosting(
        role="AI Engineer",
        required_skills=["RAG"],
        preferred_skills=[],
        responsibilities=[],
        seniority="unknown",
        keywords=[],
        raw_text_length=100,
    )
    documents = [
        ProfileDocument(
            source="projects.md",
            content="## Retrieval Project\n\nBuilt a RAG document QA system with chunk retrieval.",
        )
    ]

    matches = match_job_to_profile(parsed_job, documents, use_vector_retrieval=True)

    assert matches[0].retrieval_method == "vector"
    assert matches[0].source == "projects.md"


def test_match_job_to_profile_falls_back_when_openai_embedding_key_is_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-api-key")
    parsed_job = ParsedJobPosting(
        role="AI Engineer",
        required_skills=["Python"],
        preferred_skills=[],
        responsibilities=[],
        seniority="unknown",
        keywords=[],
        raw_text_length=100,
    )
    documents = [
        ProfileDocument(
            source="resume.md",
            content="## Skills\n\nPython",
        )
    ]

    matches = match_job_to_profile(
        parsed_job,
        documents,
        use_vector_retrieval=True,
        embedding_provider="openai",
    )

    assert matches[0].retrieval_method == "keyword"
    assert matches[0].score == 5
