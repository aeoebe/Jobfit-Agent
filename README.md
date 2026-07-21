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
- FastAPI backend and Next.js frontend
- Rule-based job posting parser
- Optional OpenAI or Ollama structured output parser with JSON Schema validation
- LangGraph workflow for parse, match, gap analysis, and report generation
- Conditional routing for learning plans vs. resume suggestions
- Human approval flag before resume suggestions are treated as approved
- Chroma vector retrieval for profile evidence search
- Local LLM provider support through Ollama
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
    api.py
  core/
    job_parser.py
    agent_workflow.py
    llm_generation.py
    llm_job_parser.py
    matcher.py
    profile_loader.py
    vector_store.py
  llm/
    providers/
      openai_provider.py
      ollama_provider.py
      factory.py
    llm_generation.py
    llm_job_parser.py
    prompts.py
    schema.py
  frontend/
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

Run the Streamlit app:

```powershell
streamlit run app/ui.py
```

Run the FastAPI backend:

```powershell
uvicorn app.api:app --reload --port 8000
```

Run the Next.js frontend:

```powershell
cd frontend
npm install
npm run dev
```

To use OpenAI:

```powershell
$env:OPENAI_API_KEY="your_api_key"
streamlit run app/ui.py
```

You can also copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

LLM_PROVIDER=openai
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

The `.env` file is ignored by git.

## Local LLM With Ollama

Install Ollama, then pull a small chat model and embedding model:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Start Ollama locally. By default, the app expects:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

In the UI or API request, use:

```text
llm_provider=ollama
model=llama3.1:8b
embedding_provider=ollama
```

The project keeps the same JSON Schema validation and fallback behavior for both OpenAI and Ollama. If the local model fails to produce valid structured JSON, the workflow falls back to rule-based or template-based output.

## Docker

Run backend and frontend:

```powershell
docker compose up --build
```

Use a host-installed Ollama from the backend container:

```text
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Or start the optional Ollama service in Compose:

```powershell
docker compose --profile ollama up --build
```

Then pull models inside the Ollama container:

```powershell
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Run tests:

```powershell
pytest
```

## Roadmap

- Add persistent vector index storage
- Add provider-level retry and structured-output repair
- Add evaluation metrics
