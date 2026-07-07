"""Dense embedding encoder for Evidence Unit text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import BATCH_SIZE, MAX_LENGTH, MODEL_NAME, resolve_device

logger = logging.getLogger(__name__)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


class EmbeddingEncoder:
    """Load a sentence-transformer model and encode text batches."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        device: str | None = None,
        batch_size: int = BATCH_SIZE,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = resolve_device(device)
        self._model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        try:
            local_path = Path(self.model_name)
            if local_path.is_dir():
                model = SentenceTransformer(
                    str(local_path),
                    device=self.device,
                    local_files_only=True,
                )
            else:
                model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

        model.max_seq_length = self.max_length
        return model

    @property
    def embedding_dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a normalized (N, D) float32 matrix."""
        if not texts:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        cleaned = [t if t and t.strip() else " " for t in texts]
        try:
            raw = self._model.encode(
                cleaned,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Embedding inference failed: {exc}") from exc

        embeddings = np.asarray(raw, dtype=np.float32)
        return normalize_embeddings(embeddings)
