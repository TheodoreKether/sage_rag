"""Configuration for dense retrieval baseline."""

from __future__ import annotations

MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.json"
DEFAULT_INDEX_DIR = "data/vector_store"
