"""Quality validation and deduplication for generated QA pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .qa_quality import is_valid_generated_pair


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass
class ValidationReport:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class QAValidator:
    """Filter empty, duplicate, malformed, and template-style QA records."""

    def __init__(self, *, strict_natural_language: bool = True) -> None:
        self.strict_natural_language = strict_natural_language
        self._seen_questions: set[str] = set()
        self._seen_qa_ids: set[str] = set()
        self._seen_qa_pairs: set[tuple[str, str]] = set()
        self._evidence_by_unit: dict[str, dict[str, str]] = {}

    def register_evidence_context(
        self,
        unit_id: str,
        *,
        evidence_text: str,
        chapter_title: str = "",
        parent_clause: str = "",
    ) -> None:
        self._evidence_by_unit[unit_id] = {
            "evidence_text": evidence_text,
            "chapter_title": chapter_title,
            "parent_clause": parent_clause,
        }

    def validate(self, record: dict[str, Any]) -> tuple[bool, str | None]:
        qa_id = record.get("qa_id", "")
        question = (record.get("question") or "").strip()
        answer = (record.get("answer") or "").strip()

        if not question:
            return False, "empty question"
        if not answer:
            return False, "empty answer"
        if not record.get("supporting_evidence"):
            return False, "missing supporting_evidence"
        if not qa_id:
            return False, "missing qa_id"

        if self.strict_natural_language:
            ctx = self._context_for_record(record)
            ok, reason = is_valid_generated_pair(
                question,
                answer,
                evidence_text=ctx.get("evidence_text", ""),
                chapter_title=ctx.get("chapter_title", ""),
                parent_clause=ctx.get("parent_clause", ""),
            )
            if not ok:
                return False, reason

        q_norm = normalize_question(question)
        if q_norm in self._seen_questions:
            return False, "duplicate question"
        pair_key = (q_norm, normalize_question(answer))
        if pair_key in self._seen_qa_pairs:
            return False, "duplicate QA pair"
        if qa_id in self._seen_qa_ids:
            return False, "duplicate qa_id"

        self._seen_questions.add(q_norm)
        self._seen_qa_pairs.add(pair_key)
        self._seen_qa_ids.add(qa_id)
        return True, None

    def _context_for_record(self, record: dict[str, Any]) -> dict[str, str]:
        support = record.get("supporting_evidence") or []
        if support and isinstance(support[0], dict):
            uid = support[0].get("unit_id", "")
            if uid in self._evidence_by_unit:
                return self._evidence_by_unit[uid]
        return {}

    def validate_batch(self, records: list[dict[str, Any]]) -> ValidationReport:
        report = ValidationReport()
        for record in records:
            ok, reason = self.validate(record)
            if ok:
                report.accepted.append(record)
            else:
                report.rejected.append(
                    {
                        "qa_id": record.get("qa_id", ""),
                        "reason": reason or "unknown",
                        "question_preview": (record.get("question") or "")[:80],
                    }
                )
        return report


def validate_candidate_pair(
    *,
    question: str,
    answer: str,
    unit_id: str,
    evidence_text: str,
    chapter_title: str = "",
    parent_clause: str = "",
) -> tuple[bool, str | None]:
    """Validate a single generated pair before record assembly."""
    return is_valid_generated_pair(
        question,
        answer,
        evidence_text=evidence_text,
        chapter_title=chapter_title,
        parent_clause=parent_clause,
    )
