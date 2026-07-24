"""SAGE-RAG v5: Expansion + Semantic Re-scoring + Risk-aware Ranking."""

from __future__ import annotations

import logging
from typing import Sequence

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.ranking.risk_aware_ranker_v5 import RiskAwareEvidenceSelectorV5
from src.sage_rag.ranking.semantic_rescorer import SemanticRescorer
from src.sage_rag.retrieval.sage_expansion_retriever import (
    SageExpansionRetriever,
    SupportsRetrieve,
)

logger = logging.getLogger(__name__)


class SageRetrieverV5(RetrieverBase):
    """BM25 → Graph Expansion → Semantic Re-score → Structure/Risk Ranking."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        *,
        semantic_rescorer: SemanticRescorer | None = None,
        ranker: RiskAwareEvidenceSelectorV5 | None = None,
        alpha: float = 0.50,
        beta: float = 0.25,
        gamma: float = 0.25,
        lam: float = 0.20,
        expansion_edge_types: Sequence[str] | None = None,
        pool_size: int | None = None,
        use_graph_expansion: bool = True,
        use_semantic_rescoring: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.graph_expander = graph_expander
        self.use_graph_expansion = bool(use_graph_expansion)
        self.use_semantic_rescoring = bool(use_semantic_rescoring)
        self.pool_size = pool_size

        self.expansion_retriever = SageExpansionRetriever(
            base_retriever,
            graph_expander,
            expansion_edge_types=expansion_edge_types,
        )

        if semantic_rescorer is not None:
            self.semantic_rescorer = semantic_rescorer
        elif isinstance(base_retriever, BM25Retriever):
            self.semantic_rescorer = SemanticRescorer(base_retriever)
        else:
            self.semantic_rescorer = None

        self.ranker = ranker or RiskAwareEvidenceSelectorV5(
            alpha=alpha, beta=beta, gamma=gamma, lam=lam
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

        if self.use_semantic_rescoring:
            if self.semantic_rescorer is None:
                raise RuntimeError(
                    "Semantic rescoring enabled but no SemanticRescorer / BM25 base"
                )
            pool = self.semantic_rescorer.rescore(query, pool)

        return self.ranker.rank(pool, query=query, top_k=top_k)
