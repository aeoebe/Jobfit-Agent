from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.job_parser import ParsedJobPosting, parse_job_posting
from llm.llm_generation import (
    generate_learning_plan_with_llm,
    generate_report_with_llm,
    generate_resume_suggestions_with_llm,
    generate_cover_letter_with_llm,
    critique_cover_letter_with_llm,
    has_real_openai_api_key
)
from llm.llm_job_parser import parse_job_posting_with_llm
from core.matcher import RequirementMatch, match_job_to_profile, summarize_matches
from core.tool_calling_matcher import match_with_tool_calling
from core.profile_loader import ProfileDocument

MAX_REVISIONS = 3
PASSING_SCORE = 75


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
    cover_letter: str
    critic_feedback: str
    critic_score: int
    revision_count: int
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
    if state.get("use_llm") and has_real_openai_api_key():
        matches = match_with_tool_calling(
            parsed_job=state["parsed_job"],
            profile_documents=state.get("profile_documents", []),
            model=state.get("model", "gpt-4o-mini"),
            embedding_provider=state.get("embedding_provider", "local"),
        )
        retrieval_name = f"tool_calling/{state.get('embedding_provider', 'local')}"
    else:
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

def draft_cover_letter_node(state: JobFitState) -> dict[str, Any]:
    revision = state.get("revision_count", 0)
    matched_requirements = [
        m.requirement
        for m in state.get("matches", [])
        if not m.gap
    ]
    
    gaps = state.get("gaps", [])
    cover_letter = ""
    used_llm = False
 
    if state.get("use_llm_generation"):
        cover_letter, used_llm = generate_cover_letter_with_llm(
            parsed_job=state["parsed_job"],
            matches=state.get("matches", []),
            gaps=gaps,
            model=state.get("model", "gpt-4o-mini"),
            critic_feedback=state.get("critic_feedback", ""),
        )
 
    if not cover_letter:
        cover_letter = _template_cover_letter(
            role=state["parsed_job"].role,
            matched=matched_requirements,
            gaps=gaps,
        )
 
    generator_name = "llm" if used_llm else "template"
    return {
        "cover_letter": cover_letter,
        "revision_count": revision + 1,
        "trace": append_trace(state, f"draft_cover_letter: revision #{revision + 1} with {generator_name}"),
    }
 
 
def critic_node(state: JobFitState) -> dict[str, Any]:
    cover_letter = state.get("cover_letter", "")
    score = PASSING_SCORE
    feedback = ""
    used_llm = False
 
    if state.get("use_llm_generation"):
        score, feedback, used_llm = critique_cover_letter_with_llm(
            parsed_job=state["parsed_job"],
            cover_letter=cover_letter,
            model=state.get("model", "gpt-4o-mini"),
        )
 
    if not used_llm:
        score = PASSING_SCORE
        feedback = "Template evaluation: skipping LLM critic."
 
    return {
        "critic_score": score,
        "critic_feedback": feedback,
        "trace": append_trace(
            state,
            f"critic: scored cover letter {score}/100 (revision #{state.get('revision_count', 0)}) with {'llm' if used_llm else 'template'}",
        ),
    }
 
def should_revise(state: JobFitState) -> str:
    score = state.get("critic_score", 0)
    revision_count = state.get("revision_count", 0)
 
    if score >= PASSING_SCORE or revision_count >= MAX_REVISIONS:
        return "generate_report"
    return "draft_cover_letter"

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
        cover_letter=state.get("cover_letter", ""),
        critic_score=state.get("critic_score"),
        revision_count=state.get("revision_count", 0),
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
    graph.add_node("draft_cover_letter", draft_cover_letter_node)
    graph.add_node("critic", critic_node)
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
    graph.add_edge("wait_for_approval", "draft_cover_letter")
    graph.add_edge("draft_cover_letter", "critic")
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "draft_cover_letter": "draft_cover_letter",
            "generate_report": "generate_report",
        },
    )
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
            "revision_count": 0,
            "trace": [],
        }
    )


def append_trace(state: JobFitState, message: str) -> list[str]:
    return [*state.get("trace", []), message]

def _template_cover_letter(
    role: str,
    matched: list[str],
    gaps: list[str],
) -> str:
    lines = [
        "Dear Hiring Manager,",
        "",
        f"I am writing to apply for the {role} position.",
        "",
    ]
    if matched:
        lines.append(f"My experience includes: {', '.join(matched[:5])}.")
        lines.append("")
    if gaps:
        lines.append(f"I am actively developing skills in: {', '.join(gaps[:3])}.")
        lines.append("")
    lines.append("I look forward to discussing how I can contribute to your team.")
    lines.append("")
    lines.append("Sincerely,")
    lines.append("[Your Name]")
 
    return "\n".join(lines)


def build_final_report(
    parsed_job: ParsedJobPosting,
    matches: list[RequirementMatch],
    gaps: list[str],
    match_summary: dict[str, Any],
    next_step: str = "",
    learning_plan: list[str] | None = None,
    resume_suggestions: list[ResumeSuggestion] | None = None,
    resume_suggestions_approved: bool = False,
    cover_letter: str = "",
    critic_score: int | None = None,
    revision_count: int = 0,
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

    if cover_letter:
        lines.extend(["", "## Cover Letter", ""])
        if critic_score is not None:
            lines.append(f"*Critic score: {critic_score}/100 — revised {revision_count} time(s)*")
            lines.append("")
        lines.append(cover_letter)

    return "\n".join(lines)


def compact_text(text: str, max_length: int = 160) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_length:
        return compacted
    return f"{compacted[: max_length - 3]}..."
