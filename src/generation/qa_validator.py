"""Quality validation and deduplication for generated QA pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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
    """Filter empty, duplicate, and malformed QA records."""

    def __init__(self) -> None:
        self._seen_questions: set[str] = set()
        self._seen_qa_ids: set[str] = set()
        self._seen_qa_pairs: set[tuple[str, str]] = set()

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
