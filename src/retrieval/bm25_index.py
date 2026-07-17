"""Build and persist BM25 indexes over Evidence Units."""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .text_tokenizer import tokenize

logger = logging.getLogger(__name__)

BM25_MODEL_FILENAME = "bm25_model.pkl"
BM25_METADATA_FILENAME = "metadata.json"
BM25_CONFIG_FILENAME = "index_config.json"

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75


@dataclass
class BM25BuildResult:
    documents: int
    output_dir: Path
    elapsed_seconds: float
    k1: float
    b: float
    avg_doc_length: float


def load_evidence_units(input_path: Path) -> list[dict[str, Any]]:
    """Load Evidence Unit records from JSONL."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Evidence units file not found: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping corrupt JSON at line %d: %s", line_no, exc)
                continue

            text = (data.get("text") or "").strip()
            unit_id = (data.get("unit_id") or "").strip()
            if not text or not unit_id:
                logger.warning("Skipping invalid record at line %d", line_no)
                continue
            records.append(data)

    if not records:
        raise ValueError(f"No valid evidence units found in {input_path}")
    return records


def build_metadata_entry(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "unit_id": record["unit_id"],
        "document_id": record["document_id"],
        "document_type": record.get("document_type", ""),
        "title": record.get("title", ""),
        "chapter_id": record.get("chapter_id", ""),
        "chapter_title": record.get("chapter_title", ""),
        "parent_clause": record.get("parent_clause", ""),
        "page": record.get("page", 0),
        "text": record.get("text", ""),
        "token_length": record.get("token_length", 0),
        "char_length": record.get("char_length", 0),
        "split_index": record.get("split_index", 1),
        "split_total": record.get("split_total", 1),
        "metadata": record.get("metadata", {}),
    }


def build_bm25_index(
    input_path: Path,
    output_dir: Path,
    *,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
) -> BM25BuildResult:
    start = time.perf_counter()
    records = load_evidence_units(input_path)

    metadata: list[dict[str, Any]] = []
    tokenized_corpus: list[list[str]] = []

    for index, record in enumerate(
        tqdm(records, desc="Tokenizing evidence units", unit="doc")
    ):
        metadata.append(build_metadata_entry(index, record))
        tokenized_corpus.append(tokenize(record["text"]))

    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    save_bm25_index(output_dir, bm25=bm25, metadata=metadata, k1=k1, b=b)

    doc_lengths = [len(doc) for doc in tokenized_corpus]
    elapsed = time.perf_counter() - start
    return BM25BuildResult(
        documents=len(records),
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        k1=k1,
        b=b,
        avg_doc_length=sum(doc_lengths) / len(doc_lengths),
    )


def save_bm25_index(
    output_dir: Path,
    *,
    bm25: BM25Okapi,
    metadata: list[dict[str, Any]],
    k1: float,
    b: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / BM25_MODEL_FILENAME
    metadata_path = output_dir / BM25_METADATA_FILENAME
    config_path = output_dir / BM25_CONFIG_FILENAME

    with model_path.open("wb") as fh:
        pickle.dump(bm25, fh)

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    config = {
        "algorithm": "BM25Okapi",
        "k1": k1,
        "b": b,
        "documents": len(metadata),
        "tokenizer": "src.retrieval.text_tokenizer.tokenize",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved BM25 index to %s (%d documents)", output_dir, len(metadata))


def load_bm25_index(index_dir: Path) -> tuple[BM25Okapi, list[dict[str, Any]], dict[str, Any]]:
    index_dir = Path(index_dir)
    model_path = index_dir / BM25_MODEL_FILENAME
    metadata_path = index_dir / BM25_METADATA_FILENAME
    config_path = index_dir / BM25_CONFIG_FILENAME

    for path in (model_path, metadata_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"BM25 index artifact not found: {path}. "
                "Run scripts/build_bm25_index.py first."
            )

    with model_path.open("rb") as fh:
        bm25 = pickle.load(fh)

    with metadata_path.open(encoding="utf-8") as fh:
        metadata = json.load(fh)

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Metadata must be a non-empty list: {metadata_path}")

    return bm25, metadata, config
