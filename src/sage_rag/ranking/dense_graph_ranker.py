"""Dense / Hybrid structure-aware ranking for the Dense+Graph experiment.

Dense variant:
  Final = α·Dense + β·Graph + γ·Coverage − λ·Risk

Hybrid variant (analysis):
  Final = 0.25·BM25 + 0.25·Dense + 0.25·Graph + 0.25·Coverage
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Literal, Sequence

from src.retrieval.retriever_base import EvidenceUnit
from src.sage_rag.ranking.risk_aware_ranker_v5 import (
    RISK_BY_RELATION,
    RISK_CROSS_DOC,
    RISK_PER_EXTRA_HOP,
    STRUCTURE_BY_RELATION_V5,
    STRUCTURE_INITIAL,
)
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2

logger = logging.getLogger(__name__)

RankMode = Literal["dense", "hybrid"]


class DenseGraphRanker:
    """Rank candidates using dense (and optional BM25) relevance + structure."""

    def __init__(
        self,
        *,
        mode: RankMode = "dense",
        alpha: float = 0.50,
        beta: float = 0.25,
        gamma: float = 0.25,
        lam: float = 0.20,
        coverage_helper: StructureRankerV2 | None = None,
    ) -> None:
        if min(alpha, beta, gamma, lam) < 0:
            raise ValueError("weights must be non-negative")
        self.mode: RankMode = mode
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
        dense = [self._get_float(c, "dense_score") for c in pool]
        bm25 = [self._get_float(c, "semantic_score") for c in pool]
        # If BM25 semantic missing, fall back to original_score min-max.
        if all(v == 0.0 for v in bm25) and any(
            (c.metadata or {}).get("original_score") is not None for c in pool
        ):
            raw = [self._fallback_retrieval(c) for c in pool]
            bm25 = self._minmax_normalize(raw)

        query_terms = self._cov._extract_query_terms(query)
        annotated: list[EvidenceUnit] = []
        for i, unit in enumerate(pool):
            struct = self._structure_score(unit)
            cov = self._cov._query_coverage_score(unit, query_terms)
            risk = self._risk_penalty(unit, coverage=cov)

            if self.mode == "hybrid":
                final = (
                    0.25 * bm25[i]
                    + 0.25 * dense[i]
                    + 0.25 * struct
                    + 0.25 * cov
                )
            else:
                final = (
                    self.alpha * dense[i]
                    + self.beta * struct
                    + self.gamma * cov
                    - self.lam * risk
                )

            annotated.append(
                self._annotate(
                    unit,
                    dense_score=dense[i],
                    bm25_score=bm25[i],
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
    def _get_float(unit: EvidenceUnit, key: str) -> float:
        meta = unit.metadata or {}
        if meta.get(key) is None:
            return 0.0
        try:
            return float(meta[key])
        except (TypeError, ValueError):
            return 0.0

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
        seed_clause = DenseGraphRanker._clause_from_unit_id(seed)
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

    def _annotate(
        self,
        unit: EvidenceUnit,
        *,
        dense_score: float,
        bm25_score: float,
        structure_score: float,
        query_coverage_score: float,
        risk_penalty: float,
        final_score: float,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["dense_score"] = float(dense_score)
        meta["bm25_score"] = float(bm25_score)
        if "semantic_score" not in meta:
            meta["semantic_score"] = float(bm25_score)
        meta["retrieval_score_normalized"] = float(dense_score)
        meta["structure_score"] = float(structure_score)
        meta["graph_score"] = float(structure_score)
        meta["structure_path_score"] = float(structure_score)
        meta["query_coverage_score"] = float(query_coverage_score)
        meta["risk_penalty"] = float(risk_penalty)
        meta["final_score"] = float(final_score)
        meta["ranker"] = f"dense_graph_ranker_{self.mode}"
        meta["graph_distance"] = self._graph_distance(unit)
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
