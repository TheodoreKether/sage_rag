"""SAGE-RAG v3: Expansion + Candidate Allocation + Structure Selection (v2)."""

from __future__ import annotations

import logging
from typing import Sequence

from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.ranking.structure_ranker_v2 import StructureRankerV2
from src.sage_rag.retrieval.candidate_allocator import (
    AllocationStrategy,
    CandidateAllocator,
)
from src.sage_rag.retrieval.sage_expansion_retriever import (
    SageExpansionRetriever,
    SupportsRetrieve,
)

logger = logging.getLogger(__name__)


class SageRetrieverV3(RetrieverBase):
    """Baseline → expand → allocate → StructureRankerV2."""

    def __init__(
        self,
        base_retriever: SupportsRetrieve,
        graph_expander: GraphExpander,
        *,
        allocator: CandidateAllocator | None = None,
        ranker: StructureRankerV2 | None = None,
        strategy: AllocationStrategy = "adaptive",
        pool_size: int | None = None,
        expansion_edge_types: Sequence[str] | None = None,
    ) -> None:
        self.expansion_retriever = SageExpansionRetriever(
            base_retriever,
            graph_expander,
            expansion_edge_types=expansion_edge_types,
        )
        self.allocator = allocator or CandidateAllocator()
        self.ranker = ranker or StructureRankerV2()
        self.strategy: AllocationStrategy = strategy
        self.pool_size = pool_size

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 10,
        pool_size: int | None = None,
        strategy: AllocationStrategy | None = None,
    ) -> list[EvidenceUnit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if initial_k <= 0:
            raise ValueError("initial_k must be positive")

        effective_pool = pool_size or self.pool_size or max(top_k * 4, initial_k + 50)
        effective_pool = max(effective_pool, top_k, initial_k)
        strat: AllocationStrategy = strategy or self.strategy

        pool = self.expansion_retriever.retrieve(
            query, top_k=effective_pool, initial_k=initial_k
        )
        allocated = self.allocator.allocate(
            pool, query=query, top_k=top_k, strategy=strat
        )
        # For none-strategy, allocated may be the full pool; still cut to top_k via v2.
        return self.ranker.rank(allocated, query=query, top_k=top_k)
