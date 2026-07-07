"""Embedding: encode Evidence Units and build FAISS vector indexes."""

from .config import BATCH_SIZE, DEVICE, MAX_LENGTH, MODEL_NAME, resolve_device
from .encoder import EmbeddingEncoder, normalize_embeddings
from .build_index import run_index_builder

__all__ = [
    "BATCH_SIZE",
    "DEVICE",
    "MAX_LENGTH",
    "MODEL_NAME",
    "EmbeddingEncoder",
    "normalize_embeddings",
    "resolve_device",
    "run_index_builder",
]
