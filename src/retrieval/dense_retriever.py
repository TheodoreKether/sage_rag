"""Dense retrieval baseline using BGE-M3 + FAISS IndexFlatIP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

try:
    from src.embedding.config import LOCAL_MODEL_PATH
    from src.embedding.encoder import EmbeddingEncoder
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.embedding.config import LOCAL_MODEL_PATH
    from src.embedding.encoder import EmbeddingEncoder

from .config import DEFAULT_INDEX_DIR, INDEX_FILENAME, METADATA_FILENAME, MODEL_NAME
from .retriever_base import EvidenceUnit, RetrieverBase

logger = logging.getLogger(__name__)


class DenseRetriever(RetrieverBase):
    """Retrieve Evidence Units by dense vector similarity (cosine via inner product)."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        *,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self._index = self._load_faiss_index()
        self._metadata = self._load_metadata()
        effective_model = model_name
        if LOCAL_MODEL_PATH and model_name == MODEL_NAME:
            effective_model = LOCAL_MODEL_PATH
        self._encoder = EmbeddingEncoder(model_name=effective_model, device=device)
        self._validate_index_metadata_alignment()

    def _load_faiss_index(self) -> faiss.Index:
        index_path = self.index_dir / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. "
                "Run src/embedding/build_index.py first."
            )
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load FAISS index from {index_path}: {exc}") from exc

        if index.ntotal == 0:
            raise ValueError(f"FAISS index is empty: {index_path}")

        logger.info("Loaded FAISS index: %d vectors, dim=%d", index.ntotal, index.d)
        return index

    def _load_metadata(self) -> list[dict[str, Any]]:
        metadata_path = self.index_dir / METADATA_FILENAME
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Run src/embedding/build_index.py first."
            )

        try:
            with metadata_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata JSON in {metadata_path}: {exc}") from exc

        if not isinstance(data, list) or not data:
            raise ValueError(f"Metadata must be a non-empty list: {metadata_path}")

        for i, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Metadata row {i} is not an object")
            for key in ("unit_id", "document_id", "text"):
                if not row.get(key):
                    raise ValueError(f"Metadata row {i} missing required field '{key}'")

        logger.info("Loaded metadata: %d records", len(data))
        return data

    def _validate_index_metadata_alignment(self) -> None:
        if self._index.ntotal != len(self._metadata):
            raise ValueError(
                f"Index/metadata size mismatch: "
                f"faiss={self._index.ntotal}, metadata={len(self._metadata)}"
            )

        indices = [row.get("index") for row in self._metadata]
        if sorted(indices) != list(range(len(self._metadata))):
            logger.warning("Metadata index fields are not strictly sequential; using list order")

    def _encode_query(self, query: str) -> np.ndarray:
        vector = self._encoder.encode([query])
        if vector.shape[0] != 1:
            raise RuntimeError("Query encoding did not return exactly one vector")
        return vector.astype(np.float32)

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

        query_vec = self._encode_query(query.strip())
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results: list[EvidenceUnit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = self._metadata[int(idx)]
            results.append(
                self._metadata_to_evidence_unit(row, rank=rank, score=float(score))
            )

        return results

    @property
    def corpus_size(self) -> int:
        return self._index.ntotal

    @property
    def known_unit_ids(self) -> set[str]:
        return {str(row["unit_id"]) for row in self._metadata}
