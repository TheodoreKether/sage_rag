"""Dense semantic re-scoring for graph-expanded candidate pools.

Reuses cached BGE-M3 document embeddings from ``data/vector_store/embeddings.npy``
(no FAISS rebuild). Query is encoded once; cosine = inner product on L2-normed vectors.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.embedding.config import LOCAL_MODEL_PATH, MODEL_NAME
from src.embedding.encoder import EmbeddingEncoder
from src.retrieval.config import DEFAULT_INDEX_DIR, METADATA_FILENAME
from src.retrieval.retriever_base import EvidenceUnit

logger = logging.getLogger(__name__)

EMBEDDINGS_FILENAME = "embeddings.npy"


class DenseRescorer:
    """Attach ``dense_score`` (pool min-max cosine) to each candidate."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        *,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        emb_path = self.index_dir / EMBEDDINGS_FILENAME
        meta_path = self.index_dir / METADATA_FILENAME
        if not emb_path.is_file():
            raise FileNotFoundError(f"Missing embedding cache: {emb_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        self._embeddings = np.load(emb_path).astype(np.float32)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("vector_store metadata.json must be a list")
        self._unit_to_idx: dict[str, int] = {}
        for i, row in enumerate(metadata):
            uid = str(row.get("unit_id") or "")
            idx = int(row.get("index", i))
            if uid:
                self._unit_to_idx[uid] = idx

        if self._embeddings.shape[0] < len(self._unit_to_idx):
            logger.warning(
                "embeddings rows (%d) < metadata units (%d)",
                self._embeddings.shape[0],
                len(self._unit_to_idx),
            )

        if encoder is not None:
            self._encoder = encoder
        else:
            effective = model_name
            if LOCAL_MODEL_PATH and model_name == MODEL_NAME:
                effective = LOCAL_MODEL_PATH
            self._encoder = EmbeddingEncoder(model_name=effective, device=device)

        logger.info(
            "DenseRescorer ready: %d cached embeddings from %s",
            self._embeddings.shape[0],
            self.index_dir,
        )

    def rescore(
        self,
        query: str,
        candidates: Sequence[EvidenceUnit],
    ) -> list[EvidenceUnit]:
        """Write ``dense_score`` / ``dense_score_raw`` onto candidate metadata."""
        if not candidates:
            return []

        raw = self._raw_cosine_scores(query, candidates)
        norm = self._minmax_normalize(raw)

        out: list[EvidenceUnit] = []
        for unit, r, n in zip(candidates, raw, norm):
            meta = copy.deepcopy(unit.metadata or {})
            meta["dense_score_raw"] = float(r)
            meta["dense_score"] = float(n)
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

    def _raw_cosine_scores(
        self, query: str, candidates: Sequence[EvidenceUnit]
    ) -> list[float]:
        q = (query or "").strip()
        if not q:
            return [0.0 for _ in candidates]

        q_vec = self._encoder.encode([q])[0].astype(np.float32)
        scores: list[float] = []
        missing_texts: list[str] = []
        missing_pos: list[int] = []

        for i, unit in enumerate(candidates):
            idx = self._unit_to_idx.get(unit.unit_id)
            if idx is not None and 0 <= idx < self._embeddings.shape[0]:
                scores.append(float(np.dot(self._embeddings[idx], q_vec)))
            else:
                scores.append(0.0)
                missing_pos.append(i)
                missing_texts.append(unit.text or "")

        if missing_texts:
            logger.debug(
                "DenseRescorer: encoding %d candidates missing from cache",
                len(missing_texts),
            )
            doc_vecs = self._encoder.encode(missing_texts)
            for pos, vec in zip(missing_pos, doc_vecs):
                scores[pos] = float(np.dot(np.asarray(vec, dtype=np.float32), q_vec))

        return scores

    @staticmethod
    def _minmax_normalize(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0 if scores[0] > 0 else 0.0]
        lo = min(scores)
        hi = max(scores)
        if hi <= lo:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - lo) / (hi - lo) for s in scores]
