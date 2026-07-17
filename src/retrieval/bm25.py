"""BM25 sparse retrieval baseline over Evidence Units."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .bm25_index import load_bm25_index
from .config import DEFAULT_BM25_INDEX_DIR
from .retriever_base import EvidenceUnit, RetrieverBase
from .text_tokenizer import tokenize

logger = logging.getLogger(__name__)


class BM25Retriever(RetrieverBase):
    """Retrieve Evidence Units via BM25Okapi over tokenized unit text."""

    def __init__(self, index_dir: str | Path = DEFAULT_BM25_INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self._bm25, self._metadata, self._config = load_bm25_index(self.index_dir)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        for i, row in enumerate(self._metadata):
            if not isinstance(row, dict):
                raise ValueError(f"Metadata row {i} is not an object")
            for key in ("unit_id", "document_id", "text"):
                if not row.get(key):
                    raise ValueError(f"Metadata row {i} missing required field '{key}'")

        logger.info(
            "Loaded BM25 index: %d documents (k1=%s, b=%s)",
            len(self._metadata),
            self._config.get("k1"),
            self._config.get("b"),
        )

    @staticmethod
    def _metadata_to_evidence_unit(
        row: dict[str, Any],
        *,
        rank: int,
        score: float,
    ) -> EvidenceUnit:
        extra = row.get("metadata") or {}
        if not isinstance(extra, dict):
            extra = {}

        return EvidenceUnit(
            rank=rank,
            score=float(score),
            unit_id=str(row["unit_id"]),
            document_id=str(row["document_id"]),
            parent_clause=str(row.get("parent_clause", "")),
            text=str(row.get("text", "")),
            metadata=extra,
            document_type=str(row.get("document_type", "")),
            title=str(row.get("title", "")),
            chapter_id=str(row.get("chapter_id", "")),
            chapter_title=str(row.get("chapter_title", "")),
            page=int(row.get("page") or 0),
            token_length=int(row.get("token_length") or 0),
            char_length=int(row.get("char_length") or 0),
            split_index=int(row.get("split_index") or 1),
            split_total=int(row.get("split_total") or 1),
        )

    def retrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_tokens = tokenize(query.strip())
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        k = min(top_k, len(scores))
        if k == 0:
            return []

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[EvidenceUnit] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            row = self._metadata[int(idx)]
            results.append(
                self._metadata_to_evidence_unit(row, rank=rank, score=score)
            )

        return results

    @property
    def corpus_size(self) -> int:
        return len(self._metadata)

    @property
    def known_unit_ids(self) -> set[str]:
        return {str(row["unit_id"]) for row in self._metadata}

    @property
    def bm25_model(self) -> BM25Okapi:
        return self._bm25
