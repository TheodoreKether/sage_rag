"""Configuration for dense retrieval baseline."""

from __future__ import annotations

MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.json"
DEFAULT_INDEX_DIR = "data/vector_store"

DEFAULT_BM25_INDEX_DIR = "data/bm25_index"
DEFAULT_BM25_RESULTS_DIR = "results/retrieval/bm25"
DEFAULT_HYBRID_RESULTS_DIR = "results/retrieval/hybrid"

DEFAULT_RRF_K = 60
DEFAULT_FUSION_TOP_N = 100
