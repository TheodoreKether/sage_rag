"""Rule-based structure-aware evidence ranking.

Combines a normalized baseline retrieval score with a graph structural prior:

    final_score = alpha * retrieval_score_normalized + beta * graph_score

No LLM / cross-encoder / trained neural reranker.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

from src.retrieval.retriever_base import EvidenceUnit

logger = logging.getLogger(__name__)

# Structural priors for expanded candidates (v1).
GRAPH_SCORE_INITIAL = 1.0
GRAPH_SCORE_BY_RELATION: dict[str, float] = {
    "parent_of": 0.8,
    "refers_to": 0.7,
    "next_to": 0.3,
}


class StructureRanker:
    """Re-rank an expansion candidate pool with retrieval + graph scores."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if abs(alpha + beta - 1.0) > 1e-6 and (alpha + beta) == 0:
            raise ValueError("alpha + beta must not both be zero")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def rank(self, candidates: Sequence[EvidenceUnit]) -> list[EvidenceUnit]:
        """Return candidates sorted by ``final_score`` (descending).

        Each returned unit gains metadata keys:
        ``retrieval_score_normalized``, ``graph_score``, ``final_score``.
        ``EvidenceUnit.score`` is set to ``final_score`` for evaluate compatibility.
        """
        if not candidates:
            return []

        scored: list[tuple[float, EvidenceUnit]] = []
        raw_scores = [self._retrieval_score(c) for c in candidates]
        normalized = self._minmax_normalize(raw_scores)

        for unit, retr_norm in zip(candidates, normalized):
            graph_score = self._graph_score(unit)
            final_score = self.alpha * retr_norm + self.beta * graph_score
            ranked_unit = self._annotate(
                unit,
                retrieval_score_normalized=retr_norm,
                graph_score=graph_score,
                final_score=final_score,
            )
            scored.append((final_score, ranked_unit))

        scored.sort(key=lambda x: (-x[0], x[1].unit_id))
        out: list[EvidenceUnit] = []
        for rank, (_, unit) in enumerate(scored, start=1):
            unit.rank = rank
            out.append(unit)
        return out

    # ------------------------------------------------------------------ scores

    @staticmethod
    def _retrieval_score(unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("original_score") is not None:
            try:
                return float(meta["original_score"])
            except (TypeError, ValueError):
                pass
        if unit.score is not None:
            return float(unit.score)
        return 0.0

    @staticmethod
    def _minmax_normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]
        lo = min(scores)
        hi = max(scores)
        if hi <= lo:
            # All equal (e.g. every expanded score is 0) → neutral 1.0
            return [1.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    @staticmethod
    def _graph_score(unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        source = meta.get("candidate_source")
        if source == "initial":
            return GRAPH_SCORE_INITIAL
        relation = meta.get("expansion_relation")
        if not relation:
            return 0.0
        return float(GRAPH_SCORE_BY_RELATION.get(str(relation), 0.0))

    @staticmethod
    def _annotate(
        unit: EvidenceUnit,
        *,
        retrieval_score_normalized: float,
        graph_score: float,
        final_score: float,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["retrieval_score_normalized"] = float(retrieval_score_normalized)
        meta["graph_score"] = float(graph_score)
        meta["final_score"] = float(final_score)
        # Keep original retrieval score discoverable.
        if "original_score" not in meta:
            meta["original_score"] = float(unit.score) if unit.score is not None else 0.0

        return EvidenceUnit(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            parent_clause=unit.parent_clause,
            text=unit.text,
            metadata=meta,
            rank=unit.rank,
            score=float(final_score),
            document_type=unit.document_type,
            title=unit.title,
            chapter_id=unit.chapter_id,
            chapter_title=unit.chapter_title,
            page=unit.page,
            token_length=unit.token_length,
            char_length=unit.char_length,
            split_index=unit.split_index,
            split_total=unit.split_total,
        )
