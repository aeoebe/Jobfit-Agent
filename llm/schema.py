from typing import Any

LEARNING_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "actions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["actions"],
}

RESUME_SUGGESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement": {"type": "string"},
                    "suggested_bullet": {"type": "string"},
                },
                "required": ["requirement", "suggested_bullet"],
            },
        }
    },
    "required": ["suggestions"],
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "markdown": {"type": "string"},
    },
    "required": ["markdown"],
}

COVER_LETTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cover_letter": {"type": "string"},
    },
    "required": ["cover_letter"],
}
 
CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "feedback": {
            "type": "string",
        },
    },
    "required": ["score", "feedback"],
}

JOB_POSTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string"},
        "required_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "preferred_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "responsibilities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "seniority": {
            "type": "string",
            "enum": ["intern", "junior", "mid", "senior", "lead", "unknown"],
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "role",
        "required_skills",
        "preferred_skills",
        "responsibilities",
        "seniority",
        "keywords",
    ],
}