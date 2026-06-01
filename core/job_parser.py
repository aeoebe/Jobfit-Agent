from __future__ import annotations

from dataclasses import asdict, dataclass
import re


SECTION_ALIASES = {
    "responsibilities": [
        "responsibilities",
        "what you will do",
        "what you'll do",
        "role responsibilities",
        "main tasks",
        "담당업무",
        "주요업무",
        "업무내용",
    ],
    "required": [
        "required qualifications",
        "requirements",
        "required skills",
        "minimum qualifications",
        "must have",
        "자격요건",
        "필수요건",
        "지원자격",
    ],
    "preferred": [
        "preferred qualifications",
        "preferred skills",
        "nice to have",
        "plus",
        "우대사항",
        "우대요건",
    ],
}

ROLE_LABELS = [
    "position",
    "role",
    "job title",
    "title",
    "채용직무",
    "모집직무",
    "직무",
    "포지션",
]

SKILL_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "apis",
    "as",
    "at",
    "build",
    "building",
    "business",
    "collaborate",
    "company",
    "create",
    "data",
    "design",
    "develop",
    "engineering",
    "experience",
    "familiarity",
    "for",
    "good",
    "have",
    "in",
    "interest",
    "internal",
    "knowledge",
    "of",
    "on",
    "or",
    "platform",
    "product",
    "products",
    "proficiency",
    "service",
    "services",
    "skill",
    "skills",
    "strong",
    "system",
    "systems",
    "team",
    "the",
    "to",
    "tool",
    "tools",
    "understanding",
    "using",
    "with",
    "work",
    "working",
}


@dataclass
class ParsedJobPosting:
    role: str
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    seniority: str
    keywords: list[str]
    raw_text_length: int

    def model_dump(self) -> dict:
        return asdict(self)


def parse_job_posting(text: str) -> ParsedJobPosting:
    """Parse a job posting into a structured object.

    This parser intentionally avoids a fixed skill dictionary. It extracts role
    and skill candidates from the posting sections themselves, so new tools and
    uncommon job titles can still appear in the result.
    """
    clean_text = normalize_text(text)
    sections = split_sections(clean_text)

    required_text = sections.get("required", "")
    preferred_text = sections.get("preferred", "")
    responsibility_text = sections.get("responsibilities", "")

    role = extract_role(clean_text, sections)
    required_skills = extract_skills(required_text or clean_text)
    preferred_skills = extract_skills(preferred_text)
    responsibilities = extract_bullets(responsibility_text)
    seniority = infer_seniority(clean_text)
    keywords = extract_keywords(clean_text, required_skills, preferred_skills)

    return ParsedJobPosting(
        role=role,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        seniority=seniority,
        keywords=keywords,
        raw_text_length=len(text),
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> dict[str, str]:
    heading_to_key = {
        heading.lower(): key
        for key, headings in SECTION_ALIASES.items()
        for heading in headings
    }

    sections: dict[str, list[str]] = {}
    current_key = "intro"

    for line in text.splitlines():
        stripped = line.strip().strip(":：")
        normalized = stripped.lower()
        matched_key = heading_to_key.get(normalized)

        if matched_key:
            current_key = matched_key
            sections.setdefault(current_key, [])
            continue

        sections.setdefault(current_key, []).append(line)

    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def extract_role(text: str, sections: dict[str, str] | None = None) -> str:
    first_lines = [line.strip() for line in text.splitlines()[:12] if line.strip()]
    first_block = "\n".join(first_lines)

    label_pattern = "|".join(re.escape(label) for label in ROLE_LABELS)
    labeled_match = re.search(
        rf"(?:{label_pattern})\s*[:：\-]\s*(.+)",
        first_block,
        flags=re.IGNORECASE,
    )
    if labeled_match:
        return clean_role(labeled_match.group(1))

    intro = (sections or {}).get("intro", "")
    intro_lines = [line.strip() for line in intro.splitlines() if line.strip()]
    for line in intro_lines + first_lines:
        if is_probable_heading(line) or is_bullet(line):
            continue
        if 2 <= len(line) <= 80:
            return clean_role(line)

    return "Unknown"


def clean_role(role: str) -> str:
    role = re.sub(r"^[\[\(【].*?[\]\)】]\s*", "", role)
    role = re.split(r"\s+[|/]\s+| - | – | — ", role, maxsplit=1)[0]
    return role.strip()


def extract_skills(text: str) -> list[str]:
    """Extract skill candidates from requirement-like text.

    The function combines two signals:
    - terms appearing after phrases such as "experience with" or "using"
    - technology-looking tokens such as Python, Next.js, PostgreSQL, RAG, C++
    """
    candidates: list[str] = []

    for bullet in extract_bullets(text):
        candidates.extend(extract_contextual_skill_phrases(bullet))
        candidates.extend(extract_technology_tokens(bullet))

    if not candidates:
        candidates.extend(extract_contextual_skill_phrases(text))
        candidates.extend(extract_technology_tokens(text))

    return unique_preserving_order(clean_skill(candidate) for candidate in candidates)


def extract_contextual_skill_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    context_pattern = re.compile(
        r"(?:experience|familiarity|proficiency|knowledge|understanding|skilled)"
        r"(?:\s+\w+){0,3}?\s+(?:with|in|of|using)\s+([^.;\n]+)",
        flags=re.IGNORECASE,
    )

    for match in context_pattern.finditer(text):
        phrase = match.group(1)
        phrases.extend(split_skill_phrase(phrase))

    return phrases


def split_skill_phrase(phrase: str) -> list[str]:
    phrase = re.sub(r"\([^)]*\)", "", phrase)
    phrase = phrase.replace(" such as ", ", ")
    phrase = phrase.replace(" including ", ", ")
    parts = re.split(r",|/|\bor\b|\band\b", phrase, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def extract_technology_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+#]*(?:\.[A-Za-z0-9]+)?\b", text)
    candidates = []

    for token in tokens:
        normalized = token.strip()
        lowered = normalized.lower()
        if lowered in SKILL_STOPWORDS:
            continue
        if looks_like_technology(normalized):
            candidates.append(normalized)

    return candidates


def looks_like_technology(token: str) -> bool:
    return (
        token.isupper()
        or token[:1].isupper()
        or any(char.isdigit() for char in token)
        or any(char in token for char in ["+", "#", "."])
        or bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", token))
        or token.endswith(("DB", "SQL", "ML", "AI", "Ops"))
    )


def clean_skill(skill: str) -> str:
    skill = skill.strip(" -•*()[]{}.,:;")
    skill = re.sub(r"\s+", " ", skill)
    skill = re.sub(
        r"^(strong|solid|hands-on|practical|deep|basic|good)\s+",
        "",
        skill,
        flags=re.IGNORECASE,
    )
    skill = re.sub(
        r"\s+(experience|knowledge|skills?|proficiency|understanding)$",
        "",
        skill,
        flags=re.IGNORECASE,
    )
    return skill.strip()


def unique_preserving_order(values: list[str] | tuple[str, ...] | object) -> list[str]:
    result = []
    seen = set()

    for value in values:
        if not value:
            continue
        key = str(value).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(str(value))

    return result


def extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        bullet = re.sub(r"^[-*•\d.)\s]+", "", stripped).strip()
        if len(bullet) >= 8:
            bullets.append(bullet)

    return bullets[:12]


def is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*[-*•\d.)]", line))


def is_probable_heading(line: str) -> bool:
    normalized = line.strip().strip(":：").lower()
    known_headings = {
        heading.lower()
        for headings in SECTION_ALIASES.values()
        for heading in headings
    }
    return normalized in known_headings


def infer_seniority(text: str) -> str:
    lowered = text.lower()

    if any(word in lowered for word in ["senior", "lead", "principal", "시니어", "리드"]):
        return "senior"
    if any(word in lowered for word in ["junior", "entry", "신입", "주니어"]):
        return "junior"
    if any(word in lowered for word in ["3+ years", "5+ years", "3년", "5년", "경력"]):
        return "mid"

    return "unknown"


def extract_keywords(text: str, required_skills: list[str], preferred_skills: list[str]) -> list[str]:
    keywords = list(dict.fromkeys(required_skills + preferred_skills))

    repeated_terms = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]{2,}\b", text)
    term_counts: dict[str, int] = {}
    for term in repeated_terms:
        key = term.lower()
        if key in SKILL_STOPWORDS:
            continue
        term_counts[term] = term_counts.get(term, 0) + 1

    for term, count in sorted(term_counts.items(), key=lambda item: item[1], reverse=True):
        if count >= 2 and term not in keywords:
            keywords.append(term)

    return keywords[:20]
