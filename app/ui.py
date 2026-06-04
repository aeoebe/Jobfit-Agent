from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from core.agent_workflow import run_jobfit_workflow
from core.profile_loader import ProfileDocument, load_profile_documents


SAMPLE_PATH = ROOT_DIR / "data" / "sample_job_posting.txt"
SAMPLE_PROFILE_PATHS = [
    ROOT_DIR / "data" / "sample_resume.md",
    ROOT_DIR / "data" / "sample_projects.md",
]


def load_sample_posting() -> str:
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_text(encoding="utf-8")
    return ""


def list_to_table(items: list[str], column_name: str) -> None:
    if items:
        st.dataframe(pd.DataFrame({column_name: items}), hide_index=True, use_container_width=True)
    else:
        st.info(f"No {column_name.lower()} found yet.")


def load_uploaded_profile_documents(uploaded_files) -> list[ProfileDocument]:
    documents = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue().decode("utf-8")
        documents.append(ProfileDocument(source=uploaded_file.name, content=content))

    return documents


st.set_page_config(
    page_title="JobFit Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("JobFit Agent")
st.caption("Paste a job posting and turn it into structured data for resume matching.")

with st.sidebar:
    st.header("Workflow")
    st.markdown(
        """
        1. Paste a job posting
        2. Parse requirements
        3. Review structured output
        4. Match against your resume later
        """
    )
    use_sample = st.toggle("Use sample posting", value=True)
    use_llm = st.toggle("Use LLM parser", value=False)
    use_llm_generation = st.toggle("Use LLM generation", value=False)
    model = st.text_input("OpenAI model", value="gpt-4o-mini", disabled=not use_llm)
    st.divider()
    use_sample_profile = st.toggle("Use sample profile", value=True)
    use_vector_retrieval = st.toggle("Use vector retrieval", value=True)
    embedding_provider = st.selectbox(
        "Embedding provider",
        options=["local", "openai"],
        index=0,
        disabled=not use_vector_retrieval,
    )
    uploaded_profile_files = st.file_uploader(
        "Upload resume/project markdown",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    approve_resume_suggestions = st.checkbox("Approve resume suggestions", value=False)

default_text = load_sample_posting() if use_sample else ""

job_posting = st.text_area(
    "Job posting",
    value=default_text,
    height=360,
    placeholder="Paste the full job posting here.",
)

analyze = st.button("Analyze posting", type="primary", use_container_width=True)

if analyze:
    if not job_posting.strip():
        st.warning("Paste a job posting first.")
        st.stop()

    profile_documents = []
    if use_sample_profile:
        profile_documents.extend(load_profile_documents(SAMPLE_PROFILE_PATHS))
    if uploaded_profile_files:
        profile_documents.extend(load_uploaded_profile_documents(uploaded_profile_files))

    with st.spinner("Running JobFit workflow..."):
        workflow_result = run_jobfit_workflow(
            job_posting=job_posting,
            profile_documents=profile_documents,
            use_llm=use_llm,
            use_vector_retrieval=use_vector_retrieval,
            use_llm_generation=use_llm_generation,
            embedding_provider=embedding_provider,
            model=model,
            resume_suggestions_approved=approve_resume_suggestions,
        )

    parsed = workflow_result["parsed_job"]
    matches = workflow_result.get("matches", [])
    match_summary = workflow_result.get("match_summary", {})
    gaps = workflow_result.get("gaps", [])
    learning_plan = workflow_result.get("learning_plan", [])
    resume_suggestions = workflow_result.get("resume_suggestions", [])

    analysis_tab, match_tab, plan_tab, suggestions_tab, trace_tab, report_tab, json_tab = st.tabs(
        ["Job Analysis", "Profile Match", "Learning Plan", "Resume Suggestions", "Agent Trace", "Report", "JSON"]
    )

    with analysis_tab:
        st.subheader("Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Role", parsed.role or "Unknown")
        summary_cols[1].metric("Seniority", parsed.seniority or "Unknown")
        summary_cols[2].metric("Skills Found", len(parsed.required_skills) + len(parsed.preferred_skills))

        left, right = st.columns(2)

        with left:
            st.subheader("Required Skills")
            list_to_table(parsed.required_skills, "Required Skill")

            st.subheader("Responsibilities")
            list_to_table(parsed.responsibilities, "Responsibility")

        with right:
            st.subheader("Preferred Skills")
            list_to_table(parsed.preferred_skills, "Preferred Skill")

            st.subheader("Keywords")
            list_to_table(parsed.keywords, "Keyword")

    with match_tab:
        if not profile_documents:
            st.warning("Add a sample profile or upload resume/project markdown files.")
        else:
            match_cols = st.columns(4)
            match_cols[0].metric("Average Score", match_summary["average_score"])
            match_cols[1].metric("Matched", match_summary["matched_count"])
            match_cols[2].metric("Gaps", match_summary["gap_count"])
            match_cols[3].metric("Total", match_summary["total_count"])

            match_rows = [match.model_dump() for match in matches]
            st.dataframe(pd.DataFrame(match_rows), hide_index=True, use_container_width=True)

            if gaps:
                st.subheader("Gap Analysis")
                list_to_table(gaps, "Missing or Weak Requirement")

    with plan_tab:
        st.subheader("Learning Plan")
        if learning_plan:
            list_to_table(learning_plan, "Recommended Action")
        else:
            st.info("No learning plan was generated for this run.")

    with suggestions_tab:
        st.subheader("Resume Suggestions")
        if resume_suggestions:
            st.caption(
                "Suggestions are only marked approved when the sidebar approval checkbox is enabled before analysis."
            )
            suggestion_rows = [suggestion.model_dump() for suggestion in resume_suggestions]
            st.dataframe(pd.DataFrame(suggestion_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No resume suggestions were generated for this run.")

    with trace_tab:
        st.subheader("Agent Workflow Trace")
        for index, event in enumerate(workflow_result.get("trace", []), start=1):
            st.write(f"{index}. {event}")

    with report_tab:
        st.subheader("Final Report")
        st.markdown(workflow_result.get("final_report", ""))

    with json_tab:
        st.subheader("Structured JSON")
        st.json(
            {
                "parsed_job": parsed.model_dump(),
                "matches": [match.model_dump() for match in matches],
                "match_summary": match_summary,
                "gaps": gaps,
                "next_step": workflow_result.get("next_step"),
                "use_vector_retrieval": workflow_result.get("use_vector_retrieval", False),
                "use_llm_generation": workflow_result.get("use_llm_generation", False),
                "embedding_provider": workflow_result.get("embedding_provider", "local"),
                "learning_plan": learning_plan,
                "resume_suggestions": [suggestion.model_dump() for suggestion in resume_suggestions],
                "resume_suggestions_approved": workflow_result.get("resume_suggestions_approved", False),
                "trace": workflow_result.get("trace", []),
            }
        )
else:
    st.info("Click Analyze posting to parse the job posting.")
