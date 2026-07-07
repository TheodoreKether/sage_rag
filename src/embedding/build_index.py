"""Build FAISS vector index from Evidence Units."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

try:
    from .config import BATCH_SIZE, DEVICE, LOCAL_MODEL_PATH, MAX_LENGTH, MODEL_NAME, resolve_device
    from .encoder import EmbeddingEncoder
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.embedding.config import BATCH_SIZE, DEVICE, LOCAL_MODEL_PATH, MAX_LENGTH, MODEL_NAME, resolve_device
    from src.embedding.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"


@dataclass
class BuildResult:
    vectors: int = 0
    dimension: int = 0
    model_name: str = ""
    device: str = ""
    elapsed_seconds: float = 0.0
    skipped_empty: int = 0
    skipped_corrupt: int = 0
    output_dir: Path = field(default_factory=Path)


def load_evidence_units(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Load evidence units from JSONL, skipping corrupt or empty-text lines."""
    jsonl_path = path
    if path.is_dir():
        jsonl_path = path / "evidence_units.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Evidence units file not found: {jsonl_path}")

    records: list[dict[str, Any]] = []
    skipped_empty = 0
    skipped_corrupt = 0

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped_corrupt += 1
                logger.warning("Skipping corrupt JSON at line %d: %s", line_no, exc)
                continue

            text = (data.get("text") or "").strip()
            if not text:
                skipped_empty += 1
                logger.warning(
                    "Skipping empty text at line %d (unit_id=%s)",
                    line_no,
                    data.get("unit_id", "?"),
                )
                continue

            if not data.get("unit_id"):
                skipped_corrupt += 1
                logger.warning("Skipping record without unit_id at line %d", line_no)
                continue

            records.append(data)

    return records, skipped_empty, skipped_corrupt


def build_metadata_entry(index: int, record: dict[str, Any]) -> dict[str, Any]:
    """Build metadata row preserving full Evidence Unit recovery fields."""
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


def encode_in_batches(
    encoder: EmbeddingEncoder,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode all texts with batched inference and a progress bar."""
    if not texts:
        return np.zeros((0, encoder.embedding_dimension), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding evidence units",
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        chunks.append(encoder.encode(batch))

    return np.vstack(chunks).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create an inner-product index for L2-normalized vectors."""
    if embeddings.size == 0:
        raise ValueError("Cannot build FAISS index from empty embedding matrix")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_vector_store(
    *,
    output_dir: Path,
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / INDEX_FILENAME
    embeddings_path = output_dir / EMBEDDINGS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    logger.info("Saved FAISS index to %s", index_path)
    logger.info("Saved embeddings to %s", embeddings_path)
    logger.info("Saved metadata to %s", metadata_path)


def render_embedding_report(result: BuildResult, *, report_path: Path) -> None:
    lines = [
        "# Embedding & Vector Index Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Number of vectors | {result.vectors} |",
        f"| Embedding dimension | {result.dimension} |",
        f"| Model | `{result.model_name}` |",
        f"| Device | {result.device} |",
        f"| Processing time | {result.elapsed_seconds:.2f} s |",
        f"| Skipped (empty text) | {result.skipped_empty} |",
        f"| Skipped (corrupt JSON) | {result.skipped_corrupt} |",
        "",
        "## Output Files",
        "",
        f"| File | Path |",
        f"|------|------|",
        f"| FAISS index | `{result.output_dir / INDEX_FILENAME}` |",
        f"| Embedding matrix | `{result.output_dir / EMBEDDINGS_FILENAME}` |",
        f"| Metadata | `{result.output_dir / METADATA_FILENAME}` |",
        "",
        "## Index Configuration",
        "",
        "- Index type: `IndexFlatIP` (inner product on L2-normalized vectors ≈ cosine similarity)",
        "- Embeddings are independent of retrieval logic and reusable across all RAG baselines.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote embedding report to %s", report_path)


def run_index_builder(
    input_path: Path,
    output_dir: Path,
    *,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    device: str | None = None,
) -> BuildResult:
    start = time.perf_counter()

    records, skipped_empty, skipped_corrupt = load_evidence_units(input_path)
    if not records:
        raise ValueError("No valid evidence units found to index")

    resolved_device = resolve_device(device)
    effective_model = model_name
    if LOCAL_MODEL_PATH and model_name == MODEL_NAME:
        effective_model = LOCAL_MODEL_PATH

    encoder = EmbeddingEncoder(
        model_name=effective_model,
        device=resolved_device,
        batch_size=batch_size,
        max_length=max_length,
    )

    texts = [r["text"] for r in records]
    embeddings = encode_in_batches(encoder, texts, batch_size=batch_size)

    metadata = [build_metadata_entry(i, r) for i, r in enumerate(records)]
    index = build_faiss_index(embeddings)
    save_vector_store(
        output_dir=output_dir,
        index=index,
        embeddings=embeddings,
        metadata=metadata,
    )

    elapsed = time.perf_counter() - start
    return BuildResult(
        vectors=len(records),
        dimension=embeddings.shape[1],
        model_name=effective_model,
        device=encoder.device,
        elapsed_seconds=elapsed,
        skipped_empty=skipped_empty,
        skipped_corrupt=skipped_corrupt,
        output_dir=output_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from Evidence Units.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/evidence_units/evidence_units.jsonl",
        help="Evidence units JSONL file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/vector_store",
        help="Output directory for FAISS index and metadata",
    )
    parser.add_argument(
        "--report",
        default="results/embedding_report.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="HuggingFace / sentence-transformers model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum sequence length for the encoder",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Device: auto, cpu, or cuda",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        result = run_index_builder(
            input_path,
            output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    render_embedding_report(result, report_path=Path(args.report))
    logger.info(
        "Done: %d vectors (dim=%d) in %.2fs on %s",
        result.vectors,
        result.dimension,
        result.elapsed_seconds,
        result.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
