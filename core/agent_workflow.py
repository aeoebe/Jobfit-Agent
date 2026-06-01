from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.job_parser import ParsedJobPosting, parse_job_posting
from core.llm_generation import (
    generate_learning_plan_with_llm,
    generate_report_with_llm,
    generate_resume_suggestions_with_llm,
)
from core.llm_job_parser import parse_job_posting_with_llm
from core.matcher import RequirementMatch, match_job_to_profile, summarize_matches
from core.profile_loader import ProfileDocument


@dataclass
class ResumeSuggestion:
    requirement: str 
    current_evidence: str
    suggested_bullet: str
    approved: bool = False

    def model_dump(self) -> dict:
        return asdict(self)


class JobFitState(TypedDict, total=False):
    job_posting: str
    profile_documents: list[ProfileDocument]
    use_llm: bool
    use_vector_retrieval: bool
    use_llm_generation: bool
    embedding_provider: str
    model: str
    parsed_job: ParsedJobPosting
    matches: list[RequirementMatch]
    match_summary: dict[str, Any]
    gaps: list[str]
    next_step: str
    learning_plan: list[str]
    resume_suggestions: list[ResumeSuggestion]
    resume_suggestions_approved: bool
    final_report: str
    trace: list[str]


def parse_job_node(state: JobFitState) -> dict[str, Any]:
    if state.get("use_llm"):
        parsed_job = parse_job_posting_with_llm(
            state["job_posting"],
            model=state.get("model", "gpt-4o-mini"),
        )
        parser_name = "llm_structured_output_parser"
    else:
        parsed_job = parse_job_posting(state["job_posting"])
        parser_name = "rule_based_parser"

    return {
        "parsed_job": parsed_job,
        "trace": append_trace(state, f"parse_job: extracted role and requirements with {parser_name}"),
    }


def match_profile_node(state: JobFitState) -> dict[str, Any]:
    matches = match_job_to_profile(
        parsed_job=state["parsed_job"],
        profile_documents=state.get("profile_documents", []),
        use_vector_retrieval=state.get("use_vector_retrieval", False),
        embedding_provider=state.get("embedding_provider", "local"),
    )
    retrieval_name = (
        f"vector/{state.get('embedding_provider', 'local')}"
        if state.get("use_vector_retrieval")
        else "keyword"
    )

    return {
        "matches": matches,
        "match_summary": summarize_matches(matches),
        "trace": append_trace(state, f"match_profile: scored {len(matches)} requirements with {retrieval_name} retrieval"),
    }


def analyze_gaps_node(state: JobFitState) -> dict[str, Any]:
    gaps = [match.requirement for match in state.get("matches", []) if match.gap]

    return {
        "gaps": gaps,
        "trace": append_trace(state, f"analyze_gaps: found {len(gaps)} weak or missing requirements"),
    }


def decide_next_step(state: JobFitState) -> str:
    gap_count = len(state.get("gaps", []))
    if gap_count >= 3:
        return "generate_learning_plan"
    return "suggest_resume_edits"


def generate_learning_plan_node(state: JobFitState) -> dict[str, Any]:
    gaps = state.get("gaps", [])
    learning_plan = []
    used_llm = False

    if state.get("use_llm_generation"):
        learning_plan, used_llm = generate_learning_plan_with_llm(
            parsed_job=state["parsed_job"],
            gaps=gaps,
            model=state.get("model", "gpt-4o-mini"),
        )

    if not learning_plan:
        learning_plan = [
            f"Build or document one small project that demonstrates {gap}."
            for gap in gaps[:5]
        ]

    if not learning_plan:
        learning_plan = ["No major gaps found. Focus on interview stories for the strongest matches."]

    generator_name = "llm" if used_llm else "template"
    return {
        "next_step": "generate_learning_plan",
        "learning_plan": learning_plan,
        "resume_suggestions": [],
        "trace": append_trace(
            state,
            f"decide_next_step: routed to generate_learning_plan because gap count is high; generated with {generator_name}",
        ),
    }


def suggest_resume_edits_node(state: JobFitState) -> dict[str, Any]:
    suggestions = []
    approved = state.get("resume_suggestions_approved", False)
    used_llm = False

    if state.get("use_llm_generation"):
        llm_suggestions, used_llm = generate_resume_suggestions_with_llm(
            parsed_job=state["parsed_job"],
            matches=state.get("matches", []),
            model=state.get("model", "gpt-4o-mini"),
        )
        if llm_suggestions:
            evidence_by_requirement = {
                match.requirement: match.matched_evidence
                for match in state.get("matches", [])
            }
            suggestions = [
                ResumeSuggestion(
                    requirement=item["requirement"],
                    current_evidence=evidence_by_requirement.get(item["requirement"], ""),
                    suggested_bullet=item["suggested_bullet"],
                    approved=approved,
                )
                for item in llm_suggestions
            ]

    if not suggestions:
        for match in state.get("matches", []):
            if match.gap:
                continue
            suggestions.append(
                ResumeSuggestion(
                    requirement=match.requirement,
                    current_evidence=match.matched_evidence,
                    suggested_bullet=(
                        f"Highlighted {match.requirement} experience using evidence from {match.source}: "
                        f"{compact_text(match.matched_evidence)}"
                    ),
                    approved=approved,
                ),
            )

    generator_name = "llm" if used_llm else "template"
    return {
        "next_step": "suggest_resume_edits",
        "resume_suggestions": suggestions[:5],
        "learning_plan": [],
        "trace": append_trace(
            state,
            f"decide_next_step: routed to suggest_resume_edits because match quality is usable; generated with {generator_name}",
        ),
    }


def wait_for_approval_node(state: JobFitState) -> dict[str, Any]:
    approved = state.get("resume_suggestions_approved", False)
    status = "approved" if approved else "waiting_for_user_approval"

    return {
        "trace": append_trace(state, f"wait_for_approval: resume suggestions are {status}"),
    }


def generate_report_node(state: JobFitState) -> dict[str, Any]:
    template_report = build_final_report(
        parsed_job=state["parsed_job"],
        matches=state.get("matches", []),
        gaps=state.get("gaps", []),
        match_summary=state.get("match_summary", {}),
        next_step=state.get("next_step", ""),
        learning_plan=state.get("learning_plan", []),
        resume_suggestions=state.get("resume_suggestions", []),
        resume_suggestions_approved=state.get("resume_suggestions_approved", False),
    )
    report = template_report
    used_llm = False

    if state.get("use_llm_generation"):
        llm_report, used_llm = generate_report_with_llm(
            template_report=template_report,
            model=state.get("model", "gpt-4o-mini"),
        )
        if llm_report:
            report = llm_report

    generator_name = "llm" if used_llm else "template"
    return {
        "final_report": report,
        "trace": append_trace(state, f"generate_report: created markdown job-fit report with {generator_name}"),
    }


def build_jobfit_graph():
    graph = StateGraph(JobFitState)
    graph.add_node("parse_job", parse_job_node)
    graph.add_node("match_profile", match_profile_node)
    graph.add_node("analyze_gaps", analyze_gaps_node)
    graph.add_node("generate_learning_plan", generate_learning_plan_node)
    graph.add_node("suggest_resume_edits", suggest_resume_edits_node)
    graph.add_node("wait_for_approval", wait_for_approval_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "parse_job")
    graph.add_edge("parse_job", "match_profile")
    graph.add_edge("match_profile", "analyze_gaps")
    graph.add_conditional_edges(
        "analyze_gaps",
        decide_next_step,
        {
            "generate_learning_plan": "generate_learning_plan",
            "suggest_resume_edits": "suggest_resume_edits",
        },
    )
    graph.add_edge("generate_learning_plan", "generate_report")
    graph.add_edge("suggest_resume_edits", "wait_for_approval")
    graph.add_edge("wait_for_approval", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_jobfit_workflow(
    job_posting: str,
    profile_documents: list[ProfileDocument],
    use_llm: bool = False,
    use_vector_retrieval: bool = False,
    use_llm_generation: bool = False,
    embedding_provider: str = "local",
    model: str = "gpt-4o-mini",
    resume_suggestions_approved: bool = False,
) -> JobFitState:
    graph = build_jobfit_graph()
    return graph.invoke(
        {
            "job_posting": job_posting,
            "profile_documents": profile_documents,
            "use_llm": use_llm,
            "use_vector_retrieval": use_vector_retrieval,
            "use_llm_generation": use_llm_generation,
            "embedding_provider": embedding_provider,
            "model": model,
            "resume_suggestions_approved": resume_suggestions_approved,
            "trace": [],
        }
    )


def append_trace(state: JobFitState, message: str) -> list[str]:
    return [*state.get("trace", []), message]


def build_final_report(
    parsed_job: ParsedJobPosting,
    matches: list[RequirementMatch],
    gaps: list[str],
    match_summary: dict[str, Any],
    next_step: str = "",
    learning_plan: list[str] | None = None,
    resume_suggestions: list[ResumeSuggestion] | None = None,
    resume_suggestions_approved: bool = False,
) -> str:
    matched = [match for match in matches if not match.gap]
    learning_plan = learning_plan or []
    resume_suggestions = resume_suggestions or []

    lines = [
        f"# JobFit Report: {parsed_job.role}",
        "",
        "## Summary",
        "",
        f"- Seniority: {parsed_job.seniority}",
        f"- Average match score: {match_summary.get('average_score', 0)}",
        f"- Matched requirements: {match_summary.get('matched_count', 0)}",
        f"- Weak or missing requirements: {match_summary.get('gap_count', 0)}",
        "",
        "## Strong Matches",
        "",
    ]

    if matched:
        for match in matched[:8]:
            lines.append(f"- {match.requirement} ({match.score}/5): {match.source}")
    else:
        lines.append("- No strong matches found yet.")

    lines.extend(["", "## Gaps", ""])
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- No major gaps found.")

    lines.extend(["", "## Next Actions", ""])
    if next_step == "generate_learning_plan":
        lines.append("- Focus on closing the largest gaps before generating application materials.")
    elif next_step == "suggest_resume_edits":
        lines.append("- Review the resume suggestions and approve them before using them in final materials.")
    elif gaps:
        lines.append("- Add or improve resume evidence for the gap requirements above.")
        lines.append("- Consider building a small project that demonstrates the highest-priority missing skill.")
    else:
        lines.append("- Turn the strongest matches into resume bullets and interview stories.")

    if learning_plan:
        lines.extend(["", "## Learning Plan", ""])
        for item in learning_plan:
            lines.append(f"- {item}")

    if resume_suggestions:
        lines.extend(["", "## Resume Suggestions", ""])
        approval_status = "approved" if resume_suggestions_approved else "not approved yet"
        lines.append(f"Approval status: {approval_status}")
        lines.append("")
        for suggestion in resume_suggestions:
            if resume_suggestions_approved:
                lines.append(f"- {suggestion.suggested_bullet}")
            else:
                lines.append(f"- Pending approval: {suggestion.requirement}")

    return "\n".join(lines)


def compact_text(text: str, max_length: int = 160) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_length:
        return compacted
    return f"{compacted[: max_length - 3]}..."
