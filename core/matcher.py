from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from core.job_parser import ParsedJobPosting
from core.profile_loader import ProfileDocument, split_profile_documents
from core.vector_store import (
    VectorSearchResult,
    VectorStoreUnavailable,
    build_profile_vector_index,
    vector_score_to_match_score,
)


@dataclass
class RequirementMatch:
    requirement: str
    requirement_type: str
    matched_evidence: str
    source: str
    score: int
    gap: bool
    retrieval_method: str = "keyword"

    def model_dump(self) -> dict:
        return asdict(self)


def match_job_to_profile(
    parsed_job: ParsedJobPosting,
    profile_documents: list[ProfileDocument],
    use_vector_retrieval: bool = False,
    embedding_provider: str = "local",
) -> list[RequirementMatch]:
    chunks = split_profile_documents(profile_documents)
    requirements = [
        ("required", skill) for skill in parsed_job.required_skills
    ] + [
        ("preferred", skill) for skill in parsed_job.preferred_skills
    ]

    vector_index = None
    if use_vector_retrieval:
        try:
            vector_index = build_profile_vector_index(profile_documents, embedding_provider=embedding_provider)
        except VectorStoreUnavailable:
            vector_index = None

    matches = []
    for requirement_type, requirement in requirements:
        if vector_index:
            vector_results = vector_index.query(requirement, top_k=1)
            matches.append(
                match_single_requirement(
                    requirement=requirement,
                    requirement_type=requirement_type,
                    chunks=chunks,
                    vector_result=vector_results[0] if vector_results else None,
                )
            )
        else:
            matches.append(
                match_single_requirement(
                    requirement=requirement,
                    requirement_type=requirement_type,
                    chunks=chunks,
                )
            )

    return matches

"""keyword matching을 쓰거나 vector search를 쓰거나가 아니라 km쓰고 점수가 높으면 pass, 충분하지 않으면 하거나 하는식으로 합치기"""
def match_single_requirement(
    requirement: str,
    requirement_type: str,
    chunks: list[ProfileDocument],
    vector_result: VectorSearchResult | None = None,
) -> RequirementMatch:
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        score = score_requirement_against_text(requirement, chunk.content)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    retrieval_method = "keyword"
    if vector_result:
        vector_score = max(
            vector_score_to_match_score(vector_result.score),
            score_requirement_against_text(requirement, vector_result.content),
        )
        if vector_score >= 3 or vector_score >= best_score:
            best_score = vector_score
            best_chunk = ProfileDocument(source=vector_result.source, content=vector_result.content)
            retrieval_method = "vector"

    final_score = min(5, best_score)
    gap = final_score < 3

    return RequirementMatch(
        requirement=requirement,
        requirement_type=requirement_type,
        matched_evidence=best_chunk.content if best_chunk and not gap else "",
        source=best_chunk.source if best_chunk and not gap else "",
        score=final_score,
        gap=gap,
        retrieval_method=retrieval_method,
    )


def score_requirement_against_text(requirement: str, text: str) -> int:
    requirement_lower = requirement.lower()
    text_lower = text.lower()

    if requirement_lower in text_lower:
        return 5

    requirement_tokens = tokenize(requirement)
    text_tokens = set(tokenize(text))
    if not requirement_tokens:
        return 0

    overlap = sum(1 for token in requirement_tokens if token in text_tokens)
    ratio = overlap / len(requirement_tokens)

    if ratio >= 0.8:
        return 4
    if ratio >= 0.5:
        return 3
    if ratio > 0:
        return 2
    return 0


def summarize_matches(matches: list[RequirementMatch]) -> dict:
    if not matches:
        return {
            "average_score": 0,
            "matched_count": 0,
            "gap_count": 0,
            "total_count": 0,
        }

    return {
        "average_score": round(sum(match.score for match in matches) / len(matches), 2),
        "matched_count": sum(1 for match in matches if not match.gap),
        "gap_count": sum(1 for match in matches if match.gap),
        "total_count": len(matches),
    }


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
        if len(token) >= 2
    ]
