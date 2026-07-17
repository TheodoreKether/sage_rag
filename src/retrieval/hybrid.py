"""Hybrid retrieval baseline combining Dense and BM25 via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from pathlib import Path

from .bm25 import BM25Retriever
from .config import (
    DEFAULT_BM25_INDEX_DIR,
    DEFAULT_FUSION_TOP_N,
    DEFAULT_INDEX_DIR,
    DEFAULT_RRF_K,
    MODEL_NAME,
)
from .dense_retriever import DenseRetriever
from .retriever_base import EvidenceUnit, RetrieverBase
from .rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever(RetrieverBase):
    """Fuse Dense and BM25 rankings with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_index_dir: str | Path = DEFAULT_INDEX_DIR,
        bm25_index_dir: str | Path = DEFAULT_BM25_INDEX_DIR,
        *,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        fusion_top_n: int = DEFAULT_FUSION_TOP_N,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.dense_index_dir = Path(dense_index_dir)
        self.bm25_index_dir = Path(bm25_index_dir)
        self.rrf_k = rrf_k
        self.fusion_top_n = fusion_top_n

        self._dense = dense_retriever or DenseRetriever(
            self.dense_index_dir,
            model_name=model_name,
            device=device,
        )
        self._bm25 = bm25_retriever or BM25Retriever(self.bm25_index_dir)

        logger.info(
            "Initialized HybridRetriever: dense=%s bm25=%s rrf_k=%d fusion_top_n=%d",
            self.dense_index_dir,
            self.bm25_index_dir,
            self.rrf_k,
            self.fusion_top_n,
        )

    @staticmethod
    def _to_fused_unit(source: EvidenceUnit, *, rank: int, fused_score: float) -> EvidenceUnit:
        return EvidenceUnit(
            rank=rank,
            score=fused_score,
            unit_id=source.unit_id,
            document_id=source.document_id,
            parent_clause=source.parent_clause,
            text=source.text,
            metadata=dict(source.metadata),
            document_type=source.document_type,
            title=source.title,
            chapter_id=source.chapter_id,
            chapter_title=source.chapter_title,
            page=source.page,
            token_length=source.token_length,
            char_length=source.char_length,
            split_index=source.split_index,
            split_total=source.split_total,
        )

    def retrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        candidate_k = max(self.fusion_top_n, top_k)
        dense_hits = self._dense.retrieve(query, candidate_k)
        bm25_hits = self._bm25.retrieve(query, candidate_k)

        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            k=self.rrf_k,
        )

        results: list[EvidenceUnit] = []
        for rank, (unit_id, fused_score, source) in enumerate(fused[:top_k], start=1):
            results.append(
                self._to_fused_unit(source, rank=rank, fused_score=fused_score)
            )
        return results

    @property
    def dense_retriever(self) -> DenseRetriever:
        return self._dense

    @property
    def bm25_retriever(self) -> BM25Retriever:
        return self._bm25

    @property
    def known_unit_ids(self) -> set[str]:
        return self._dense.known_unit_ids | self._bm25.known_unit_ids
