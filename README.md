# JobFit Agent

JobFit Agent is a portfolio project for practicing practical AI agent workflows around job preparation.

The first version focuses on one small but useful flow:

1. Paste a job posting.
2. Parse the posting into structured data.
3. Show required skills, preferred skills, responsibilities, and a rough seniority estimate.
4. Match requirements against resume and project documents.
5. Use the match result later for cover letter drafting and interview preparation.

## Why This Project

I have experience building LLM applications and want to build a more agentic system that includes tool use, state management, human approval, evaluation, and traceability.

This project starts simple and grows toward an agent workflow:

- Job posting parser
- Candidate profile retrieval
- Requirement-to-experience matching
- Resume improvement suggestions
- Cover letter generation
- Interview question generation
- Human-in-the-loop approval
- Evaluation for groundedness and job alignment

## Current Features

- Streamlit UI for pasting a job posting
- Rule-based job posting parser
- Optional OpenAI structured output parser with JSON Schema validation
- LangGraph workflow for parse, match, gap analysis, and report generation
- Conditional routing for learning plans vs. resume suggestions
- Human approval flag before resume suggestions are treated as approved
- Chroma vector retrieval for profile evidence search
- Sample resume and project documents
- Requirement-to-profile matching with evidence and gap detection
- Structured output as JSON
- Skills and responsibilities displayed as tables
- Sample job posting for quick testing

## Project Structure

```text
jobfit-agent/
  app/
    ui.py
  core/
    job_parser.py
    agent_workflow.py
    llm_generation.py
    llm_job_parser.py
    matcher.py
    profile_loader.py
    vector_store.py
  data/
    sample_job_posting.txt
    sample_projects.md
    sample_resume.md
  tests/
    test_job_parser.py
    test_agent_workflow.py
    test_llm_job_parser.py
    test_matcher.py
    test_profile_loader.py
    test_vector_store.py
  README.md
  requirements.txt
```

## Getting Started

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app/ui.py
```

To use the LLM parser:

```powershell
$env:OPENAI_API_KEY="your_api_key"
streamlit run app/ui.py
```

You can also copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

The `.env` file is ignored by git.

Run tests:

```powershell
pytest
```

## Roadmap

- Add vector search for candidate evidence
- Add FastAPI backend
- Add LangGraph or OpenAI Agents SDK workflow
- Add FastAPI backend
- Add trace view for agent decisions
- Add evaluation metrics
