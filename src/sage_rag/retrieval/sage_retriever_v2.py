"""SAGE-RAG v2: expansion + StructureRankerV2 greedy evidence selection."""

from __future__ import annotations

import logging
from typing import Sequence

from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2
from src.sage_rag.retrieval.sage_expansion_retriever import (
    SageExpansionRetriever,
    SupportsRetrieve,
)

logger = logging.getLogger(__name__)


class SageRetrieverV2(RetrieverBase):
    """Baseline + Graph Expansion + Structure-aware Evidence Selection (v2)."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        ranker: StructureRankerV2 | None = None,
        *,
        alpha: float = 0.40,
        beta: float = 0.20,
        gamma: float = 0.45,
        lam: float = 0.35,
        expansion_edge_types: Sequence[str] | None = None,
        pool_size: int | None = None,
    ) -> None:
        self.expansion_retriever = SageExpansionRetriever(
            base_retriever,
            graph_expander,
            expansion_edge_types=expansion_edge_types,
        )
        self.ranker = ranker or StructureRankerV2(
            alpha=alpha, beta=beta, gamma=gamma, lam=lam
        )
        self.pool_size = pool_size

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 10,
        pool_size: int | None = None,
    ) -> list[EvidenceUnit]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if initial_k <= 0:
            raise ValueError("initial_k must be a positive integer")

        effective_pool = pool_size or self.pool_size or max(top_k * 4, initial_k + 50)
        effective_pool = max(effective_pool, top_k, initial_k)

        pool = self.expansion_retriever.retrieve(
            query,
            top_k=effective_pool,
            initial_k=initial_k,
        )
        return self.ranker.rank(pool, query=query, top_k=top_k)
