from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from core.job_parser import ParsedJobPosting
from core.matcher import RequirementMatch, score_requirement_against_text
from core.profile_loader import ProfileDocument
from core.vector_store import VectorStoreUnavailable, build_profile_vector_index, vector_score_to_match_score

from llm.prompts import EVALUATION_PROMPT


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_resume_vector_store",
            "description": (
                "Search the candidate's resume and project documents for evidence "
                "related to a specific skill or requirement. Use this for every "
                "requirement you consider important in the job posting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search query describing the skill or experience "
                            "to look for, e.g. 'Kubernetes container orchestration experience'"
                        ),
                    },
                    "requirement": {
                        "type": "string",
                        "description": (
                            "The exact requirement string from the job posting, "
                            "e.g. 'Kubernetes'"
                        ),
                    },
                    "requirement_type": {
                        "type": "string",
                        "enum": ["required", "preferred"],
                        "description": "Whether this is a required or preferred skill.",
                    },
                },
                "required": ["query", "requirement", "requirement_type"],
            },
        },
    }
]

MAX_TOOL_CALLS = 20
TOP_K = 3

def match_with_tool_calling(
    parsed_job: ParsedJobPosting,
    profile_documents: list[ProfileDocument],
    model: str = "gpt-4o-mini",
    embedding_provider: str = "local",
) -> list[RequirementMatch]:
    try:
        vector_index = build_profile_vector_index(profile_documents, embedding_provider)
    except VectorStoreUnavailable:
        vector_index = None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a recruiter assistant analyzing a job posting. "
                "Use the search tool to find evidence for each requirement you consider important. "
                "Search all required skills first, then preferred skills. "
                "Call the tool once per requirement — do not skip any required skill. "
                "If a direct search returns weak evidence, try a broader or related query "
                "(e.g. if 'Kubernetes' is weak, also try 'container orchestration' or 'EKS')."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analyze this job posting and search the candidate's resume for each requirement.\n\n"
                f"Role: {parsed_job.role}\n"
                f"Seniority: {parsed_job.seniority}\n"
                f"Required skills: {parsed_job.required_skills}\n"
                f"Preferred skills: {parsed_job.preferred_skills}\n"
                f"Responsibilities: {parsed_job.responsibilities}\n\n"
                f"Search for every required skill and any preferred skills you think are important."
            ),
        },
    ]

    raw_evidence: dict[str, list[dict[str, Any]]] = {}
    searched: dict[str, RequirementMatch] = {}

    for _ in range(MAX_TOOL_CALLS):
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message


        if not msg.tool_calls:
            break

        messages.append(msg)

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args["query"]
            requirement = args["requirement"]
            requirement_type = args.get("requirement_type", "required")

            best_result = None
            if vector_index:
                results = vector_index.query(query, top_k=TOP_K)
                if results:
                    best_result = max(
                        results,
                        key=lambda r: vector_score_to_match_score(r.score),
                    )

            match = _build_match(
                requirement=requirement,
                requirement_type=requirement_type,
                vector_result=best_result,
            )

            if requirement not in searched or match.score > searched[requirement].score:
                searched[requirement] = match

            if requirement not in raw_evidence:
                raw_evidence[requirement] = []
            if best_result:
                raw_evidence[requirement].append({
                    "query": query,
                    "content": best_result.content,
                    "source": best_result.source,
                    "vector_score": best_result.score,
                })

            tool_result = {
                "requirement": requirement,
                "found": not match.gap,
                "score": match.score,
                "evidence_preview": match.matched_evidence[:200] if match.matched_evidence else "",
                "source": match.source,
            }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

    if searched:
        return _llm_final_evaluation(
            client=client,
            model=model,
            parsed_job=parsed_job,
            searched=searched,
            raw_evidence=raw_evidence,
        )

    return list(searched.values())


def _llm_final_evaluation(
    client: OpenAI,
    model: str,
    parsed_job: ParsedJobPosting,
    searched: dict[str, RequirementMatch],
    raw_evidence: dict[str, list[dict[str, Any]]],
) -> list[RequirementMatch]:
    evidence_summary = []
    for requirement, match in searched.items():
        evidence_list = raw_evidence.get(requirement, [])
        evidence_summary.append({
            "requirement": requirement,
            "requirement_type": match.requirement_type,
            "current_score": match.score,
            "current_gap": match.gap,
            "evidence_found": [
                {"query": e["query"], "content": e["content"][:300], "source": e["source"]}
                for e in evidence_list
            ],
        })

    evaluation_prompt = EVALUATION_PROMPT.format(
        role=parsed_job.role,
        seniority=parsed_job.seniority,
        evidence=json.dumps(evidence_summary, indent=2)
    ).strip()

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior recruiter. Respond only with the requested JSON array.",
                },
                {"role": "user", "content": evaluation_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        evaluations: list[dict] = json.loads(content.strip())
    except Exception:
        return list(searched.values())

    final_matches = []
    eval_by_requirement = {e["requirement"]: e for e in evaluations}

    for requirement, match in searched.items():
        eval_result = eval_by_requirement.get(requirement)
        if eval_result:
            new_score = min(5, max(0, int(eval_result.get("score", match.score))))
            new_gap = bool(eval_result.get("gap", match.gap))
            final_matches.append(RequirementMatch(
                requirement=requirement,
                requirement_type=match.requirement_type,
                matched_evidence=match.matched_evidence if not new_gap else "",
                source=match.source if not new_gap else "",
                score=new_score,
                gap=new_gap,
                retrieval_method="tool_calling+llm_eval",
            ))
        else:
            final_matches.append(match)

    return final_matches


def _build_match(
    requirement: str,
    requirement_type: str,
    vector_result,
) -> RequirementMatch:
    if vector_result is None:
        return RequirementMatch(
            requirement=requirement,
            requirement_type=requirement_type,
            matched_evidence="",
            source="",
            score=0,
            gap=True,
            retrieval_method="tool_calling",
        )

    vector_score = vector_score_to_match_score(vector_result.score)
    keyword_score = score_requirement_against_text(requirement, vector_result.content)
    final_score = min(5, max(vector_score, keyword_score))
    gap = final_score < 3

    return RequirementMatch(
        requirement=requirement,
        requirement_type=requirement_type,
        matched_evidence=vector_result.content if not gap else "",
        source=vector_result.source if not gap else "",
        score=final_score,
        gap=gap,
        retrieval_method="tool_calling",
    )