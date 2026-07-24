"""Structure-aware Candidate Allocation (SAGE v3).

Sits between Graph Expansion and StructureRankerV2:

    expansion pool → allocate(strategy) → evidence selection (v2)

Strategies (paper ablation):
- ``none``:     no reservation (pass-through pool)
- ``fixed``:    reserve a fixed number of graph slots (e.g. 7+3)
- ``adaptive``: grow/shrink graph budget from expanded quality
"""

from __future__ import annotations

import copy
import logging
from typing import Literal, Sequence

from src.retrieval.retriever_base import EvidenceUnit
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2

logger = logging.getLogger(__name__)

AllocationStrategy = Literal["none", "fixed", "adaptive"]


class CandidateAllocator:
    """Budget graph-expanded evidence into the final competition set."""

    def __init__(
        self,
        *,
        fixed_graph_budget: int = 3,
        adaptive_default_budget: int = 2,
        adaptive_high_budget: int = 4,
        path_threshold: float = 0.7,
        coverage_threshold: float = 0.25,
        scorer: StructureRankerV2 | None = None,
    ) -> None:
        self.fixed_graph_budget = int(fixed_graph_budget)
        self.adaptive_default_budget = int(adaptive_default_budget)
        self.adaptive_high_budget = int(adaptive_high_budget)
        self.path_threshold = float(path_threshold)
        self.coverage_threshold = float(coverage_threshold)
        # Reuse v2 path/coverage definitions for consistent ablation.
        self._scorer = scorer or StructureRankerV2()

    def allocate(
        self,
        candidates: Sequence[EvidenceUnit],
        query: str,
        top_k: int = 10,
        strategy: AllocationStrategy = "adaptive",
    ) -> list[EvidenceUnit]:
        """Return an allocated candidate set for downstream selection.

        For ``fixed`` / ``adaptive`` (hard slot reservation):
            originals: ``top_k - graph_budget``
            graph:     up to ``graph_budget`` highest-priority expanded
        Unfilled graph slots fall back to additional originals.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        strategy = strategy.lower()  # type: ignore[assignment]
        if strategy not in ("none", "fixed", "adaptive"):
            raise ValueError(f"Unknown strategy: {strategy}")

        originals, expanded = self._split(candidates)
        if strategy == "none":
            return [
                self._tag(u, allocation_source="passthrough", graph_budget=0)
                for u in candidates
            ]

        budget = (
            self.fixed_graph_budget
            if strategy == "fixed"
            else self._adaptive_budget(expanded, query)
        )
        budget = max(0, min(budget, top_k))
        original_budget = top_k - budget

        ranked_expanded = self._rank_expanded(expanded, query)
        reserved_graph = ranked_expanded[:budget]

        ranked_original = self._rank_original(originals)
        kept_original = ranked_original[:original_budget]

        # Fill unused graph slots with remaining originals.
        shortfall = budget - len(reserved_graph)
        if shortfall > 0:
            kept_original = (
                kept_original
                + ranked_original[original_budget : original_budget + shortfall]
            )

        allocated: list[EvidenceUnit] = []
        seen: set[str] = set()
        for u in kept_original:
            if u.unit_id in seen:
                continue
            seen.add(u.unit_id)
            allocated.append(
                self._tag(u, allocation_source="original", graph_budget=budget)
            )
        for u in reserved_graph:
            if u.unit_id in seen:
                continue
            seen.add(u.unit_id)
            allocated.append(
                self._tag(u, allocation_source="graph_reserved", graph_budget=budget)
            )

        logger.debug(
            "allocate strategy=%s budget=%d originals=%d graph=%d → %d",
            strategy,
            budget,
            sum(1 for u in allocated if (u.metadata or {}).get("allocation_source") == "original"),
            sum(
                1
                for u in allocated
                if (u.metadata or {}).get("allocation_source") == "graph_reserved"
            ),
            len(allocated),
        )
        return allocated

    # ------------------------------------------------------------------ helpers

    def _adaptive_budget(self, expanded: list[EvidenceUnit], query: str) -> int:
        """Grow budget when HQ expanded exist; shrink when they do not.

        - ≥2 high-quality expanded → ``adaptive_high_budget`` (default 4)
        - 1 high-quality expanded → ``adaptive_default_budget`` (default 2)
        - none high-quality → 1 if any expanded else 0 (reduce vs default)
        """
        if not expanded:
            return 0
        query_terms = self._scorer._extract_query_terms(query)
        n_hq = 0
        for u in expanded:
            path = self._scorer._structure_path_score(u)
            cov = self._scorer._query_coverage_score(u, query_terms)
            if path >= self.path_threshold and cov >= self.coverage_threshold:
                n_hq += 1
        if n_hq >= 2:
            return self.adaptive_high_budget
        if n_hq >= 1:
            return self.adaptive_default_budget
        # 否则减少：仍保留最小 1 槽，便于 ablation 观察 graph 进入情况
        return 1

    def _rank_expanded(
        self, expanded: list[EvidenceUnit], query: str
    ) -> list[EvidenceUnit]:
        query_terms = self._scorer._extract_query_terms(query)
        scored: list[tuple[float, EvidenceUnit]] = []
        for u in expanded:
            path = self._scorer._structure_path_score(u)
            cov = self._scorer._query_coverage_score(u, query_terms)
            # Prefer high path ∧ coverage; relation prior already in path.
            priority = 0.55 * path + 0.45 * cov
            scored.append((priority, u))
        scored.sort(key=lambda x: (-x[0], x[1].unit_id))
        return [u for _, u in scored]

    @staticmethod
    def _rank_original(originals: list[EvidenceUnit]) -> list[EvidenceUnit]:
        def key(u: EvidenceUnit) -> tuple[float, str]:
            meta = u.metadata or {}
            try:
                score = float(meta.get("original_score", u.score or 0.0))
            except (TypeError, ValueError):
                score = float(u.score or 0.0)
            return (-score, u.unit_id)

        return sorted(originals, key=key)

    @staticmethod
    def _split(
        candidates: Sequence[EvidenceUnit],
    ) -> tuple[list[EvidenceUnit], list[EvidenceUnit]]:
        originals: list[EvidenceUnit] = []
        expanded: list[EvidenceUnit] = []
        for u in candidates:
            src = (u.metadata or {}).get("candidate_source")
            if src == "expanded":
                expanded.append(u)
            else:
                originals.append(u)
        return originals, expanded

    @staticmethod
    def _tag(
        unit: EvidenceUnit,
        *,
        allocation_source: str,
        graph_budget: int,
    ) -> EvidenceUnit:
        meta = copy.deepcopy(unit.metadata or {})
        meta["allocation_source"] = allocation_source
        meta["graph_budget"] = int(graph_budget)
        return EvidenceUnit(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            parent_clause=unit.parent_clause,
            text=unit.text,
            metadata=meta,
            rank=unit.rank,
            score=unit.score,
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
