"""Natural-language quality checks for retrieval-oriented QA generation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# Phrases that leak document structure — forbidden in questions.
FORBIDDEN_PHRASE_PATTERNS: tuple[str, ...] = (
    r"根据条款",
    r"依据条款",
    r"按照\s*[\d\.]+",
    r"本标准规定",
    r"本标准参考",
    r"本标准",
    r"本文件规定",
    r"标准\s*[\w\-]+\s*条款",
    r"第\s*\d+\s*[章节条]",
    r"第\s*[一二三四五六七八九十百千]+\s*[章节条]",
    r"条款\s*[\d\.]+",
    r"章节\s*[\d\.]+",
    r"表\s*\d+",
    r"Table\s+\d+",
    r"Annex\s+[A-Z]",
    r"附录\s*[A-Z]?",
    r"\bClause\s+[\d\.]+",
    r"\bSection\s+[\d\.]+",
    r"按照第",
    r"按照标准",
)

CLAUSE_NUMBER_PATTERN = re.compile(
    r"(?:"
    r"条款\s*[\d\.]+|"
    r"第\s*\d+(?:\.\d+)*\s*[章节条]?|"
    r"[\d]+\.[\d]+(?:\.[\d]+)*|"
    r"表\s*\d+|Table\s+\d+"
    r")",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[。！？；.!?;])\s+", cleaned, maxsplit=1)
    return parts[0].strip()


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def check_natural_question(
    question: str,
    *,
    evidence_text: str,
    chapter_title: str = "",
    parent_clause: str = "",
) -> tuple[bool, str | None]:
    """Return (ok, rejection_reason)."""
    q = (question or "").strip()
    if not q:
        return False, "empty question"

    for pattern in FORBIDDEN_PHRASE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return False, f"forbidden phrase pattern: {pattern}"

    if CLAUSE_NUMBER_PATTERN.search(q):
        return False, "contains clause/section/table number"

    if parent_clause and parent_clause in q:
        return False, "mentions parent_clause id"

    if chapter_title and len(chapter_title) >= 4:
        if _similarity(q, chapter_title) >= 0.72:
            return False, "copies chapter title"
        if chapter_title in q:
            return False, "mentions chapter title"

    first_sent = _first_sentence(evidence_text)
    if first_sent and len(first_sent) >= 12:
        if _similarity(q, first_sent) >= 0.65:
            return False, "copies first sentence of evidence"
        opening = first_sent[: min(20, len(first_sent))]
        if opening in q and len(q) <= len(first_sent) + 20:
            return False, "copies evidence opening"

    # Evidence heading often appears as first token block before content
    heading = evidence_text.strip().split("\n", 1)[0].strip()
    if len(heading) >= 6 and len(heading) <= 80:
        q_stripped = q.strip().rstrip("？?")
        if q_stripped == heading:
            return False, "copies evidence heading"
        if _similarity(q, heading) >= 0.88:
            return False, "copies evidence heading"
        # Reject only when the question is essentially the heading with minimal rephrasing.
        if heading in q and len(q) <= len(heading) + 12:
            return False, "copies evidence heading"

    return True, None


def is_valid_generated_pair(
    question: str,
    answer: str,
    *,
    evidence_text: str,
    chapter_title: str = "",
    parent_clause: str = "",
) -> tuple[bool, str | None]:
    if not (answer or "").strip():
        return False, "empty answer"
    return check_natural_question(
        question,
        evidence_text=evidence_text,
        chapter_title=chapter_title,
        parent_clause=parent_clause,
    )
