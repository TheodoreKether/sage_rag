"""BM25 generation + Graph Expansion + Dense (or Hybrid) structure ranking."""

from __future__ import annotations

import logging
from typing import Sequence

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.ranking.dense_graph_ranker import DenseGraphRanker, RankMode
from src.sage_rag.ranking.dense_rescorer import DenseRescorer
from src.sage_rag.ranking.semantic_rescorer import SemanticRescorer
from src.sage_rag.retrieval.sage_expansion_retriever import (
    SageExpansionRetriever,
    SupportsRetrieve,
)

logger = logging.getLogger(__name__)


class SageDenseGraphRetriever(RetrieverBase):
    """BM25 → Expand → Dense rescore → Structure/Risk ranking."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        *,
        dense_rescorer: DenseRescorer,
        bm25_rescorer: SemanticRescorer | None = None,
        ranker: DenseGraphRanker | None = None,
        mode: RankMode = "dense",
        alpha: float = 0.50,
        beta: float = 0.25,
        gamma: float = 0.25,
        lam: float = 0.20,
        expansion_edge_types: Sequence[str] | None = None,
        pool_size: int | None = None,
        use_graph_expansion: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.graph_expander = graph_expander
        self.dense_rescorer = dense_rescorer
        self.use_graph_expansion = bool(use_graph_expansion)
        self.pool_size = pool_size
        self.mode: RankMode = mode

        self.expansion_retriever = SageExpansionRetriever(
            base_retriever,
            graph_expander,
            expansion_edge_types=expansion_edge_types,
        )

        if bm25_rescorer is not None:
            self.bm25_rescorer = bm25_rescorer
        elif isinstance(base_retriever, BM25Retriever):
            self.bm25_rescorer = SemanticRescorer(base_retriever)
        else:
            self.bm25_rescorer = None

        self.ranker = ranker or DenseGraphRanker(
            mode=mode, alpha=alpha, beta=beta, gamma=gamma, lam=lam
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 20,
        pool_size: int | None = None,
    ) -> list[EvidenceUnit]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if initial_k <= 0:
            raise ValueError("initial_k must be a positive integer")

        effective_pool = pool_size or self.pool_size or max(top_k * 5, initial_k + 50)
        effective_pool = max(effective_pool, top_k, initial_k)

        if self.use_graph_expansion:
            pool = self.expansion_retriever.retrieve(
                query, top_k=effective_pool, initial_k=initial_k
            )
        else:
            hits = self.base_retriever.retrieve(query, top_k=initial_k)
            pool = [SageExpansionRetriever._mark_initial(u) for u in hits]

        # Always attach dense scores; BM25 semantic optional (needed for hybrid).
        if self.bm25_rescorer is not None:
            pool = self.bm25_rescorer.rescore(query, pool)
        pool = self.dense_rescorer.rescore(query, pool)
        return self.ranker.rank(pool, query=query, top_k=top_k)
