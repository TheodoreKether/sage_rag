"""Semantic re-scoring for SAGE v5 candidate pools.

All candidates (initial + expanded) receive the same query–evidence
relevance score. v1 uses corpus BM25 (no training / no LLM).
"""

from __future__ import annotations

import copy
import logging
from typing import Sequence

import numpy as np

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever_base import EvidenceUnit
from src.retrieval.text_tokenizer import tokenize

logger = logging.getLogger(__name__)


class SemanticRescorer:
    """Recompute BM25(query, evidence) for every candidate in the pool."""

    def __init__(self, bm25_retriever: BM25Retriever) -> None:
        self.bm25 = bm25_retriever
        self._unit_to_idx: dict[str, int] = {
            str(row["unit_id"]): i for i, row in enumerate(self.bm25._metadata)
        }

    def rescore(
        self,
        query: str,
        candidates: Sequence[EvidenceUnit],
    ) -> list[EvidenceUnit]:
        """Attach ``semantic_score`` (pool min-max in [0, 1]) to each candidate.

        Also stores ``semantic_score_raw`` (corpus BM25 score) for analysis.
        Initial and expanded candidates use the **same** scoring path.
        """
        if not candidates:
            return []

        raw_scores = self._raw_bm25_scores(query, candidates)
        norm = self._minmax_normalize(raw_scores)

        out: list[EvidenceUnit] = []
        for unit, raw, sem in zip(candidates, raw_scores, norm):
            meta = copy.deepcopy(unit.metadata or {})
            meta["semantic_score_raw"] = float(raw)
            meta["semantic_score"] = float(sem)
            out.append(
                EvidenceUnit(
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
            )
        return out

    def _raw_bm25_scores(
        self, query: str, candidates: Sequence[EvidenceUnit]
    ) -> list[float]:
        q = (query or "").strip()
        if not q:
            return [0.0 for _ in candidates]
        tokens = tokenize(q)
        if not tokens:
            return [0.0 for _ in candidates]

        corpus_scores = self.bm25.bm25_model.get_scores(tokens)
        scores: list[float] = []
        missing = 0
        for unit in candidates:
            idx = self._unit_to_idx.get(unit.unit_id)
            if idx is None:
                missing += 1
                scores.append(0.0)
                continue
            scores.append(float(corpus_scores[idx]))
        if missing:
            logger.debug(
                "SemanticRescorer: %d/%d candidates missing from BM25 index",
                missing,
                len(candidates),
            )
        return scores

    @staticmethod
    def _minmax_normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0 if scores[0] > 0 else 0.0]
        arr = np.asarray(scores, dtype=float)
        lo = float(arr.min())
        hi = float(arr.max())
        if hi <= lo:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [float((s - lo) / (hi - lo)) for s in scores]
