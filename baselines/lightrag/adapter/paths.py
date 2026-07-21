"""Path constants for the LightRAG baseline workspace."""

from __future__ import annotations

from pathlib import Path

# baselines/lightrag/
BASELINE_ROOT = Path(__file__).resolve().parents[1]
# sage_rag/
REPO_ROOT = BASELINE_ROOT.parents[1]

OFFICIAL_ROOT = BASELINE_ROOT / "LightRAG"
RAG_STORAGE_DIR = BASELINE_ROOT / "rag_storage"
MAPS_DIR = BASELINE_ROOT / "maps"
RESULTS_DIR = BASELINE_ROOT / "results"
ENV_FILE = BASELINE_ROOT / ".env"

CHUNK_TO_UNIT_MAP = MAPS_DIR / "chunk_id_to_unit_id.json"
UNIT_TO_CHUNK_MAP = MAPS_DIR / "unit_id_to_chunk_id.json"
INGEST_MANIFEST = MAPS_DIR / "ingest_manifest.json"

EVIDENCE_UNITS = REPO_ROOT / "data" / "evidence_units" / "evidence_units.jsonl"
QA_CLEAN = REPO_ROOT / "data" / "qa_dataset" / "qa_pairs_clean.jsonl"

# Optional paper-facing mirror (thin copy of final metrics)
PAPER_RESULTS_DIR = REPO_ROOT / "results" / "retrieval" / "lightrag"
