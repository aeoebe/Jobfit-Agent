LEARNING_PLAN_PROMPT = """
    Create a concise learning plan for a candidate applying to this role.

    Role: {role}
    Seniority: {seniority}
    Gaps: {gaps}

    Return 3-5 practical actions. Each action should be specific and portfolio-oriented.
"""

RESUME_SUGGESTION_PROMPT = """
    Suggest resume bullets for this job application using only the provided evidence.

    Role: {role}
    Responsibilities: {responsibilities}
    Evidence matches: {matches}

    Rules:
    - Do not invent metrics, tools, employers, or outcomes.
    - Each bullet must be grounded in one evidence item.
    - Keep bullets concise and action-oriented.
"""

GENERATE_REPORT_PROMPT = """
    Refine this markdown job-fit report for clarity and usefulness.

    Rules:
    - Preserve the same factual content.
    - Do not invent new experience, scores, or evidence.
    - Keep markdown headings.
    - Keep it concise.

    Report:
    {template}
"""

GENERATE_COVER_LETTER_PROMPT = """
    Write a professional cover letter for this job application.

    Role: {role}
    Seniority: {seniority}

    Evidence:
    {evidence}

    Gaps:
    {gaps}

    {feedback}

    Rules:
    - Use only the evidence provided.
    - Do not invent metrics, employers, technologies, or outcomes.
    - Connect the evidence to the job requirements.
    - Mention the strongest evidence first.
    - Keep it under 400 words.
    - Be specific and action-oriented.
 """

CRITIQUE_COVER_LETTER_PROMPT = """
    Evaluate this cover letter against the job requirements.
    
    Required skills: {required_skills}
    Preferred skills: {preferred_skills}
    Responsibilities: {responsibilities}
    
    Cover letter:
    {cv}
    
    Evaluation rules:
    - Check if required skills are addressed.
    - Check if preferred skills are mentioned.
    - Penalize invented facts or vague claims.
    - Be specific about what is missing, referencing skill names directly.
"""

JOB_PARSING_PROMPT = """
    You extract structured information from job postings.

    Rules:
    - Extract only information supported by the job posting text.
    - Do not invent skills, tools, responsibilities, or seniority.
    - Keep skill names concise, for example "Python", "LangGraph", "RAG", "Vector DB".
    - Put must-have qualifications in required_skills.
    - Put nice-to-have or preferred qualifications in preferred_skills.
    - If seniority is not explicit, use "unknown".
    - Return the result using the provided JSON schema.
"""

EVALUATION_PROMPT = """
    You are a senior recruiter making final hiring assessments.

    Job Role: {role} ({seniority})

    Below are requirements from the job posting with evidence collected from the candidate's resume.
    Your task: re-evaluate each requirement's match score (0-5) and gap status.

    Scoring guide:
    - 5: Direct, strong evidence (exact skill mentioned with context)
    - 4: Clear related evidence (e.g. EKS experience for Kubernetes requirement)
    - 3: Partial evidence (adjacent skills, transferable experience)
    - 2: Weak or indirect evidence
    - 0-1: No meaningful evidence → gap=true

    Important: Use your judgment to recognize equivalent or transferable experience.
    For example: Docker Swarm or EKS can satisfy a Kubernetes requirement at score 3-4.

    Requirements and evidence:
    {evidence}

    Respond with a JSON array only, no extra text:
    [
    {{
        "requirement": "<requirement string>",
        "score": <0-5 integer>,
        "gap": <true|false>,
        "reasoning": "<one sentence explaining your decision>"
    }},
    ...
    ]
"""