"""SAGE v5 risk-aware ranking over semantically re-scored candidates.

FinalScore =
    α · SemanticScore
  + β · StructureScore
  + γ · CoverageScore
  − λ · RiskScore

Unlike v4, SemanticScore comes from independent BM25(query, evidence)
re-scoring (same for initial and expanded) — not seed-score inheritance.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Sequence

from src.retrieval.retriever_base import EvidenceUnit
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2

logger = logging.getLogger(__name__)

# v5 relation priors (next_to down, refers_to up vs v4).
STRUCTURE_BY_RELATION_V5: dict[str, float] = {
    "parent_of": 1.0,
    "refers_to": 0.9,
    "next_to": 0.3,
}
STRUCTURE_INITIAL = 1.0

RISK_BY_RELATION: dict[str, float] = {
    "next_to": 0.4,
    "parent_of": 0.1,
    "refers_to": 0.15,
}
RISK_CROSS_DOC = 0.3
RISK_PER_EXTRA_HOP = 0.1


class RiskAwareEvidenceSelectorV5:
    """Flat ranking using semantic_score + structure + coverage − risk."""

    def __init__(
        self,
        alpha: float = 0.50,
        beta: float = 0.25,
        gamma: float = 0.25,
        lam: float = 0.20,
        *,
        coverage_helper: StructureRankerV2 | None = None,
    ) -> None:
        if min(alpha, beta, gamma, lam) < 0:
            raise ValueError("weights must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self._cov = coverage_helper or StructureRankerV2()

    def rank(
        self,
        candidates: Sequence[EvidenceUnit],
        query: str = "",
        top_k: int | None = None,
    ) -> list[EvidenceUnit]:
        scored = self.score_candidates(candidates, query=query)
        if top_k is not None:
            scored = scored[: max(1, min(int(top_k), len(scored)))]
        for rank, unit in enumerate(scored, start=1):
            unit.rank = rank
        return scored

    def score_candidates(
        self,
        candidates: Sequence[EvidenceUnit],
        query: str = "",
    ) -> list[EvidenceUnit]:
        if not candidates:
            return []

        pool = list(candidates)
        # Prefer metadata semantic_score; fall back to original_score min-max.
        sem_raw = [self._semantic_score(c) for c in pool]
        if any(s is not None for s in sem_raw):
            filled = [0.0 if s is None else float(s) for s in sem_raw]
            # Already pool-normalized by SemanticRescorer; keep as-is.
            semantic = filled
        else:
            fallback = [self._fallback_retrieval(c) for c in pool]
            semantic = self._minmax_normalize(fallback)

        query_terms = self._cov._extract_query_terms(query)
        annotated: list[EvidenceUnit] = []
        for i, unit in enumerate(pool):
            struct = self._structure_score(unit)
            cov = self._cov._query_coverage_score(unit, query_terms)
            risk = self._risk_penalty(unit, coverage=cov)
            final = (
                self.alpha * semantic[i]
                + self.beta * struct
                + self.gamma * cov
                - self.lam * risk
            )
            annotated.append(
                self._annotate(
                    unit,
                    semantic_score=semantic[i],
                    structure_score=struct,
                    query_coverage_score=cov,
                    risk_penalty=risk,
                    final_score=final,
                )
            )

        annotated.sort(
            key=lambda u: (
                -float((u.metadata or {}).get("final_score", u.score or 0.0)),
                u.unit_id,
            )
        )
        return annotated

    @staticmethod
    def _semantic_score(unit: EvidenceUnit) -> float | None:
        meta = unit.metadata or {}
        if meta.get("semantic_score") is None:
            return None
        try:
            return float(meta["semantic_score"])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fallback_retrieval(unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("original_score") is not None:
            try:
                return float(meta["original_score"])
            except (TypeError, ValueError):
                pass
        if unit.score is not None:
            try:
                return float(unit.score)
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def _minmax_normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0 if scores[0] > 0 else 0.0]
        lo = min(scores)
        hi = max(scores)
        if hi <= lo:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def _structure_score(self, unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("candidate_source") == "initial":
            return STRUCTURE_INITIAL
        relation = str(meta.get("expansion_relation") or "")
        base = float(STRUCTURE_BY_RELATION_V5.get(relation, 0.0))
        if base <= 0:
            return 0.0
        dist = self._graph_distance(unit)
        return max(0.0, base * math.exp(-float(max(0, dist - 1))))

    def _risk_penalty(self, unit: EvidenceUnit, *, coverage: float = 0.0) -> float:
        meta = unit.metadata or {}
        if meta.get("candidate_source") != "expanded":
            return 0.0
        relation = str(meta.get("expansion_relation") or "")
        risk = float(RISK_BY_RELATION.get(relation, 0.25))
        if self._is_cross_document(unit):
            risk += RISK_CROSS_DOC
        dist = self._graph_distance(unit)
        if dist > 1:
            risk += RISK_PER_EXTRA_HOP * (dist - 1)
        risk = max(0.0, min(1.0, risk))
        cov = max(0.0, min(1.0, float(coverage)))
        return max(0.0, min(1.0, risk * (1.0 - cov)))

    @staticmethod
    def _is_cross_document(unit: EvidenceUnit) -> bool:
        meta = unit.metadata or {}
        seed = str(meta.get("expanded_from") or "")
        if not seed:
            return False
        seed_doc = seed.split("::", 1)[0]
        cand_doc = (unit.document_id or "").strip()
        if not seed_doc or not cand_doc:
            return False
        return seed_doc != cand_doc

    @staticmethod
    def _graph_distance(unit: EvidenceUnit) -> int:
        meta = unit.metadata or {}
        if meta.get("candidate_source") == "initial":
            return 0
        if meta.get("graph_distance") is not None:
            try:
                return max(1, int(meta["graph_distance"]))
            except (TypeError, ValueError):
                pass
        seed = str(meta.get("expanded_from") or "")
        cand_clause = str(unit.parent_clause or "")
        seed_clause = RiskAwareEvidenceSelectorV5._clause_from_unit_id(seed)
        if seed_clause and cand_clause:
            if cand_clause == seed_clause:
                return 1
            if seed_clause.startswith(cand_clause + ".") or cand_clause.startswith(
                seed_clause + "."
            ):
                depth_gap = abs(seed_clause.count(".") - cand_clause.count("."))
                return max(1, depth_gap)
        return 1

    @staticmethod
    def _clause_from_unit_id(unit_id: str) -> str:
        parts = unit_id.split("::")
        if len(parts) >= 3:
            return parts[2]
        return ""

    @staticmethod
    def _annotate(
        unit: EvidenceUnit,
        *,
        semantic_score: float,
        structure_score: float,
        query_coverage_score: float,
        risk_penalty: float,
        final_score: float,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["semantic_score"] = float(semantic_score)
        meta["retrieval_score_normalized"] = float(semantic_score)
        meta["structure_score"] = float(structure_score)
        meta["graph_score"] = float(structure_score)
        meta["structure_path_score"] = float(structure_score)
        meta["query_coverage_score"] = float(query_coverage_score)
        meta["risk_penalty"] = float(risk_penalty)
        meta["final_score"] = float(final_score)
        meta["ranker"] = "risk_aware_ranker_v5"
        meta["graph_distance"] = RiskAwareEvidenceSelectorV5._graph_distance(unit)
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
