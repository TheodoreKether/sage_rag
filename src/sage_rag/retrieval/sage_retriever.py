"""End-to-end SAGE-RAG retriever: expansion + structure-aware ranking.

Pipeline::

    query
      → SageExpansionRetriever  (baseline + graph expansion, no re-rank)
      → StructureRanker.rank()  (retrieval⊕graph score)
      → final top_k EvidenceUnit list

Compatible with future ``evaluate_retrieval`` hooks via ``retrieve(query, top_k)``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.ranking.structure_ranker import StructureRanker
from src.sage_rag.retrieval.sage_expansion_retriever import (
    SageExpansionRetriever,
    SupportsRetrieve,
)

logger = logging.getLogger(__name__)


class SageRetriever(RetrieverBase):
    """Baseline + Graph Expansion + Structure-aware Ranking."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        ranker: StructureRanker | None = None,
        *,
        alpha: float = 0.7,
        beta: float = 0.3,
        expansion_edge_types: Sequence[str] | None = None,
        pool_size: int | None = None,
    ) -> None:
        self.expansion_retriever = SageExpansionRetriever(
            base_retriever,
            graph_expander,
            expansion_edge_types=expansion_edge_types,
        )
        self.ranker = ranker if ranker is not None else StructureRanker(alpha=alpha, beta=beta)
        self.pool_size = pool_size

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        pool_size: int | None = None,
    ) -> list[EvidenceUnit]:
        """Expand a candidate pool, then structure-rank and cut to ``top_k``.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Number of results after ranking.
        initial_k:
            Baseline seed count for expansion.
        pool_size:
            Max expansion-pool size before ranking. Defaults to
            ``max(top_k * 3, initial_k + 40)`` (or constructor ``pool_size``).
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if initial_k <= 0:
            raise ValueError("initial_k must be a positive integer")

        effective_pool = pool_size or self.pool_size or max(top_k * 3, initial_k + 40)
        effective_pool = max(effective_pool, top_k, initial_k)

        pool = self.expansion_retriever.retrieve(
            query,
            top_k=effective_pool,
            initial_k=initial_k,
        )
        ranked = self.ranker.rank(pool)
        final = ranked[:top_k]
        for rank, unit in enumerate(final, start=1):
            unit.rank = rank
        return final
