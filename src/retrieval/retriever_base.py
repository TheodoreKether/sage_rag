"""Abstract retriever interface for comparable RAG retrieval baselines."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceUnit:
    """A retrieved Evidence Unit with optional ranking metadata."""

    unit_id: str
    document_id: str
    parent_clause: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    score: float | None = None
    document_type: str = ""
    title: str = ""
    chapter_id: str = ""
    chapter_title: str = ""
    page: int = 0
    token_length: int = 0
    char_length: int = 0
    split_index: int = 1
    split_total: int = 1

    def to_result_dict(self) -> dict[str, Any]:
        """Serialize to the standard retrieval result schema."""
        return {
            "rank": self.rank,
            "score": self.score,
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "parent_clause": self.parent_clause,
            "text": self.text,
            "metadata": self.metadata,
        }


class RetrieverBase(abc.ABC):
    """Backend-agnostic interface: query -> ranked Evidence Units."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        """Return the top-k most relevant Evidence Units for a query."""

    @staticmethod
    def results_as_dicts(units: list[EvidenceUnit]) -> list[dict[str, Any]]:
        return [unit.to_result_dict() for unit in units]
