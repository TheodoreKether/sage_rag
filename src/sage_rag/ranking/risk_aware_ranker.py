"""Risk-aware Structure-aware Evidence Selection (SAGE v4).

Score(e | q) =
    α · RetrievalScore(e)
  + β · StructureScore(e)
  + γ · QueryCoverage(e)
  − λ · RiskPenalty(e)

Unlike v3 fixed/adaptive allocation, **all** candidates compete in one pool.
Graph-expanded evidence replaces an original only when its net score wins —
structure noise is controlled by an explicit RiskPenalty.

Design notes (rule-based, no training / LLM):
- RetrievalScore: min-max over raw BM25/Dense; expanded inherit
  ``w(relation) · seed_retrieval`` (original_score metadata stays 0).
- StructureScore: relation prior × ``exp(-(distance-1))`` (1-hop keeps prior).
- QueryCoverage: reused from StructureRankerV2 (jieba + CJK n-grams).
- RiskPenalty: relation / cross-doc / distance risk, then discounted by
  ``(1 - coverage)`` so only query-grounded expansions pay low risk.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Sequence

from src.retrieval.retriever_base import EvidenceUnit
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2

logger = logging.getLogger(__name__)

# Structure priors (before distance decay).
STRUCTURE_BY_RELATION: dict[str, float] = {
    "parent_of": 1.0,
    "refers_to": 0.85,
    "next_to": 0.5,
}
STRUCTURE_INITIAL = 1.0

# Base relation risk (Rule 1–3).
RISK_BY_RELATION: dict[str, float] = {
    "next_to": 0.4,
    "parent_of": 0.1,
    "refers_to": 0.15,
}
RISK_CROSS_DOC = 0.3
RISK_PER_EXTRA_HOP = 0.1


class RiskAwareEvidenceSelector:
    """Flat risk-aware ranking over the expansion candidate pool."""

    def __init__(
        self,
        alpha: float = 0.45,
        beta: float = 0.25,
        gamma: float = 0.30,
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
        # Reuse v2 query-coverage / term extraction (no LLM, no training).
        self._cov = coverage_helper or StructureRankerV2()

    def rank(
        self,
        candidates: Sequence[EvidenceUnit],
        query: str = "",
        top_k: int | None = None,
    ) -> list[EvidenceUnit]:
        """Score all candidates, sort descending, return Top-k."""
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
        """Return all candidates annotated & sorted by final score (desc)."""
        if not candidates:
            return []

        pool = list(candidates)
        raw = [self._retrieval_score(c) for c in pool]
        # Min-max over *raw BM25/Dense scores* (expanded raw stays 0).
        base_norm = self._minmax_normalize(raw)
        id_to_norm = {pool[i].unit_id: base_norm[i] for i in range(len(pool))}

        # Effective retrieval: initials use base_norm; expanded inherit a
        # discounted seed score (rule-based, no training).
        retr_eff = [
            self._effective_retrieval(pool[i], base_norm[i], id_to_norm)
            for i in range(len(pool))
        ]

        query_terms = self._cov._extract_query_terms(query)

        annotated: list[EvidenceUnit] = []
        for i, unit in enumerate(pool):
            struct = self._structure_score(unit)
            cov = self._cov._query_coverage_score(unit, query_terms)
            risk = self._risk_penalty(unit, coverage=cov)
            final = (
                self.alpha * retr_eff[i]
                + self.beta * struct
                + self.gamma * cov
                - self.lam * risk
            )
            annotated.append(
                self._annotate(
                    unit,
                    retrieval_score_normalized=retr_eff[i],
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

    # ------------------------------------------------------------------ scores

    # Seed-retrieval inheritance weights by relation (≤ parent prior).
    _SEED_RETR_WEIGHT: dict[str, float] = {
        "parent_of": 0.50,
        "refers_to": 0.38,
        "next_to": 0.10,
    }

    def _effective_retrieval(
        self,
        unit: EvidenceUnit,
        base_norm: float,
        id_to_norm: dict[str, float],
    ) -> float:
        """Retrieval term used in Score.

        - initial: min-max normalized original score
        - expanded: ``w(relation) · seed_retrieval`` (original_score stays 0 in meta)
        """
        meta = unit.metadata or {}
        if meta.get("candidate_source") != "expanded":
            return float(base_norm)
        relation = str(meta.get("expansion_relation") or "")
        w = float(self._SEED_RETR_WEIGHT.get(relation, 0.10))
        seed = str(meta.get("expanded_from") or "")
        seed_retr = float(id_to_norm.get(seed, 0.0))
        return max(0.0, min(1.0, w * seed_retr))

    @staticmethod
    def _retrieval_score(unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        src = meta.get("candidate_source")
        # Expanded with no retrieval score → 0 (explicit).
        if src == "expanded":
            if meta.get("original_score") is not None:
                try:
                    return float(meta["original_score"])
                except (TypeError, ValueError):
                    return 0.0
            return 0.0
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
            return [1.0]
        lo = min(scores)
        hi = max(scores)
        if hi <= lo:
            # All equal (e.g. all expanded zeros + identical BM25) → 1 for non-zero, 0 for zero
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def _structure_score(self, unit: EvidenceUnit) -> float:
        meta = unit.metadata or {}
        if meta.get("candidate_source") == "initial":
            return STRUCTURE_INITIAL
        relation = str(meta.get("expansion_relation") or "")
        base = float(STRUCTURE_BY_RELATION.get(relation, 0.0))
        if base <= 0:
            return 0.0
        dist = self._graph_distance(unit)
        # Decay on *extra* hops: 1-hop keeps the relation prior.
        # (Raw exp(-distance) would map parent_of@1 → 0.37 vs initial 1.0,
        #  making expanded evidence almost never competitive under α/β defaults.)
        return max(0.0, base * math.exp(-float(max(0, dist - 1))))

    def _risk_penalty(self, unit: EvidenceUnit, *, coverage: float = 0.0) -> float:
        """Structural expansion risk in [0, 1]. Initial candidates → 0.

        Raw relation/cross-doc/distance risk is **discounted by query coverage**:
        high coverage ⇒ expansion is query-grounded ⇒ lower effective risk.
        This is the risk-aware gate: unreliable (low-coverage) neighbors pay full
        penalty; reliable ones may compete with BM25 tails.
        """
        meta = unit.metadata or {}
        if meta.get("candidate_source") != "expanded":
            return 0.0

        relation = str(meta.get("expansion_relation") or "")
        risk = float(RISK_BY_RELATION.get(relation, 0.25))

        # Rule 4: cross-document expansion.
        if self._is_cross_document(unit):
            risk += RISK_CROSS_DOC

        # Rule 5: farther hops → higher risk.
        dist = self._graph_distance(unit)
        if dist > 1:
            risk += RISK_PER_EXTRA_HOP * (dist - 1)

        risk = max(0.0, min(1.0, risk))
        cov = max(0.0, min(1.0, float(coverage)))
        # Reliability discount (core risk-aware mechanism).
        effective = max(0.0, min(1.0, risk * (1.0 - cov)))
        return effective

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
        # Infer from clause numbering vs expansion seed (same as v2).
        seed = str(meta.get("expanded_from") or "")
        cand_clause = str(unit.parent_clause or "")
        seed_clause = RiskAwareEvidenceSelector._clause_from_unit_id(seed)
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
        retrieval_score_normalized: float,
        structure_score: float,
        query_coverage_score: float,
        risk_penalty: float,
        final_score: float,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["retrieval_score_normalized"] = float(retrieval_score_normalized)
        meta["structure_score"] = float(structure_score)
        meta["graph_score"] = float(structure_score)
        meta["structure_path_score"] = float(structure_score)
        meta["query_coverage_score"] = float(query_coverage_score)
        meta["risk_penalty"] = float(risk_penalty)
        meta["final_score"] = float(final_score)
        meta["ranker"] = "risk_aware_ranker_v4"
        meta["graph_distance"] = RiskAwareEvidenceSelector._graph_distance(unit)
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
