from core.agent_workflow import run_jobfit_workflow
from core.profile_loader import ProfileDocument


def test_run_jobfit_workflow_routes_to_resume_suggestions_when_gap_count_is_low():
    job_posting = """
AI Agent Engineer

Requirements:
- Strong Python experience.
- Experience with LangGraph.
"""
    profile_documents = [
        ProfileDocument(
            source="resume.md",
            content="## Skills\n\nPython\n\n## Projects\n\nBuilt a RAG document QA prototype.",
        )
    ]

    result = run_jobfit_workflow(
        job_posting,
        profile_documents,
        use_vector_retrieval=False,
        resume_suggestions_approved=True,
    )

    assert result["parsed_job"].role == "AI Agent Engineer"
    assert result["matches"]
    assert result["gaps"]
    assert result["next_step"] == "suggest_resume_edits"
    assert result["resume_suggestions"]
    assert result["resume_suggestions"][0].approved
    assert result["trace"] == [
        "parse_job: extracted role and requirements with rule_based_parser",
        "match_profile: scored 2 requirements with keyword retrieval",
        "analyze_gaps: found 1 weak or missing requirements",
        "decide_next_step: routed to suggest_resume_edits because match quality is usable",
        "wait_for_approval: resume suggestions are approved",
        "generate_report: created markdown job-fit report",
    ]
    assert "# JobFit Report" in result["final_report"]
    assert "Approval status: approved" in result["final_report"]


def test_run_jobfit_workflow_routes_to_learning_plan_when_gap_count_is_high():
    job_posting = """
AI Agent Engineer

Requirements:
- Strong Python experience.
- Experience with LangGraph.
- Experience with Chroma.
- Experience with Docker.
"""
    profile_documents = [
        ProfileDocument(
            source="resume.md",
            content="## Skills\n\nPython",
        )
    ]

    result = run_jobfit_workflow(job_posting, profile_documents)

    assert result["next_step"] == "generate_learning_plan"
    assert result["learning_plan"]
    assert not result["resume_suggestions"]
    assert "Learning Plan" in result["final_report"]
