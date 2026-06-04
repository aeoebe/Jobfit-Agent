from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.agent_workflow import run_jobfit_workflow
from core.job_parser import parse_job_posting
from llm.llm_job_parser import parse_job_posting_with_llm
from core.matcher import match_job_to_profile, summarize_matches
from core.profile_loader import ProfileDocument, load_profile_documents
from core.vector_store import VectorStoreUnavailable, build_profile_vector_index

app = FastAPI(
    title="JobFit Agent API",
    description="FastAPI backend for the JobFit Agent workflow.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_PROFILE_PATHS = [
    ROOT_DIR / "data" / "sample_resume.md",
    ROOT_DIR / "data" / "sample_projects.md",
]


def _load_sample_profiles() -> list[ProfileDocument]:
    return load_profile_documents(SAMPLE_PROFILE_PATHS)


@app.get(
    "/sample/posting",
    summary="return sample job posting texts",
    tags=["Sample"],
)
def get_sample_posting() -> dict:
    path = ROOT_DIR / "data" / "sample_job_posting.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample posting file not found.")
    return {"content": path.read_text(encoding="utf-8")}


class ParseRequest(BaseModel):
    job_posting: str
    use_llm: bool = False
    model: str = "gpt-4o-mini"


@app.post(
    "/parse",
    summary="Job Posting Parsing (LLM or rule-based)",
    tags=["Parse"],
)
def parse(req: ParseRequest) -> dict:
    if req.use_llm:
        parsed = parse_job_posting_with_llm(req.job_posting, model=req.model)
    else:
        parsed = parse_job_posting(req.job_posting)
    return parsed.model_dump()


class MatchRequest(BaseModel):
    job_posting: str
    use_llm: bool = False
    use_vector_retrieval: bool = False
    embedding_provider: str = "local"
    model: str = "gpt-4o-mini"
    use_sample_profile: bool = True


@app.post(
    "/match",
    summary="Profile Matching",
    tags=["Match"],
)
def match(req: MatchRequest) -> dict:
    if req.use_llm:
        parsed = parse_job_posting_with_llm(req.job_posting, model=req.model)
    else:
        parsed = parse_job_posting(req.job_posting)

    profile_documents: list[ProfileDocument] = []
    if req.use_sample_profile:
        profile_documents = _load_sample_profiles()

    matches = match_job_to_profile(
        parsed_job=parsed,
        profile_documents=profile_documents,
        use_vector_retrieval=req.use_vector_retrieval,
        embedding_provider=req.embedding_provider,
    )
    return {
        "parsed_job": parsed.model_dump(),
        "matches": [m.model_dump() for m in matches],
        "summary": summarize_matches(matches),
    }


class AnalyzeRequest(BaseModel):
    job_posting: str
    use_llm: bool = False
    use_vector_retrieval: bool = False
    use_llm_generation: bool = False
    embedding_provider: str = "local"
    model: str = "gpt-4o-mini"
    use_sample_profile: bool = True
    resume_suggestions_approved: bool = False


def _run_analyze(
    job_posting: str,
    use_llm: bool,
    use_vector_retrieval: bool,
    use_llm_generation: bool,
    embedding_provider: str,
    model: str,
    use_sample_profile: bool,
    resume_suggestions_approved: bool,
    extra_documents: list[ProfileDocument] | None = None,
) -> dict:
    profile_documents: list[ProfileDocument] = []
    if use_sample_profile:
        profile_documents = _load_sample_profiles()
    if extra_documents:
        profile_documents.extend(extra_documents)

    result = run_jobfit_workflow(
        job_posting=job_posting,
        profile_documents=profile_documents,
        use_llm=use_llm,
        use_vector_retrieval=use_vector_retrieval,
        use_llm_generation=use_llm_generation,
        embedding_provider=embedding_provider,
        model=model,
        resume_suggestions_approved=resume_suggestions_approved,
    )
    return {
        "parsed_job": result["parsed_job"].model_dump(),
        "matches": [m.model_dump() for m in result.get("matches", [])],
        "match_summary": result.get("match_summary", {}),
        "gaps": result.get("gaps", []),
        "next_step": result.get("next_step", ""),
        "learning_plan": result.get("learning_plan", []),
        "resume_suggestions": [s.model_dump() for s in result.get("resume_suggestions", [])],
        "final_report": result.get("final_report", ""),
        "trace": result.get("trace", []),
    }


@app.post(
    "/analyze",
    summary="Run LangGraph workflow",
    tags=["Analyze"],
)
def analyze(req: AnalyzeRequest) -> dict:
    return _run_analyze(
        job_posting=req.job_posting,
        use_llm=req.use_llm,
        use_vector_retrieval=req.use_vector_retrieval,
        use_llm_generation=req.use_llm_generation,
        embedding_provider=req.embedding_provider,
        model=req.model,
        use_sample_profile=req.use_sample_profile,
        resume_suggestions_approved=req.resume_suggestions_approved,
    )


@app.post(
    "/analyze/upload",
    summary="Run entire workflow",
    tags=["Analyze"],
)
async def analyze_with_upload(
    job_posting: str = Form(...),
    use_llm: bool = Form(False),
    use_vector_retrieval: bool = Form(False),
    use_llm_generation: bool = Form(False),
    embedding_provider: str = Form("local"),
    model: str = Form("gpt-4o-mini"),
    use_sample_profile: bool = Form(True),
    resume_suggestions_approved: bool = Form(False),
    files: list[UploadFile] = File(default=[]),
) -> dict:
    extra_documents: list[ProfileDocument] = []
    for file in files:
        content = (await file.read()).decode("utf-8")
        extra_documents.append(ProfileDocument(source=file.filename, content=content))

    return _run_analyze(
        job_posting=job_posting,
        use_llm=use_llm,
        use_vector_retrieval=use_vector_retrieval,
        use_llm_generation=use_llm_generation,
        embedding_provider=embedding_provider,
        model=model,
        use_sample_profile=use_sample_profile,
        resume_suggestions_approved=resume_suggestions_approved,
        extra_documents=extra_documents,
    )


class VectorSearchRequest(BaseModel):
    query: str
    use_sample_profile: bool = True
    embedding_provider: str = "local"
    top_k: int = 3


@app.post(
    "/vector/search",
    summary="Vector search",
    tags=["Vector"],
)
def vector_search(req: VectorSearchRequest) -> dict:
    profile_documents: list[ProfileDocument] = []
    if req.use_sample_profile:
        profile_documents = _load_sample_profiles()

    if not profile_documents:
        raise HTTPException(status_code=400, detail="No profile documents available to search.")

    try:
        index = build_profile_vector_index(
            profile_documents,
            embedding_provider=req.embedding_provider,
        )
        results = index.query(req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "results": [
                {"content": r.content, "source": r.source, "score": r.score}
                for r in results
            ],
        }
    except VectorStoreUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get(
    "/vector/status",
    summary="Checking Vector Store (ChromaDB) Availability",
    tags=["Vector"],
)
def vector_status() -> dict:
    try:
        profile_documents = _load_sample_profiles()
        index = build_profile_vector_index(profile_documents)
        return {
            "available": True,
            "chunk_count": len(index.chunks),
            "sources": list({chunk.source for chunk in index.chunks}),
        }
    except VectorStoreUnavailable as e:
        return {"available": False, "reason": str(e)}
    except Exception as e:
        return {"available": False, "reason": str(e)}