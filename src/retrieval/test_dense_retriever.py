"""CLI for testing the dense retrieval baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from .config import DEFAULT_INDEX_DIR, DEFAULT_TOP_K, MODEL_NAME
    from .dense_retriever import DenseRetriever
    from .retriever_base import EvidenceUnit
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.retrieval.config import DEFAULT_INDEX_DIR, DEFAULT_TOP_K, MODEL_NAME
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.retriever_base import EvidenceUnit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test dense retrieval over FAISS vector store (BGE-M3 + IndexFlatIP).",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_DIR,
        help="Vector store directory (faiss.index + metadata.json)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of evidence units to retrieve",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Query encoder model name or local path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for query encoding: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--qa-file",
        default=None,
        help="Optional QA JSONL; run sample questions for smoke testing",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of questions to sample from --qa-file",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="Characters to show in text preview",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def format_hit(hit: EvidenceUnit, *, preview_chars: int) -> str:
    preview = hit.text.replace("\n", " ").strip()
    if len(preview) > preview_chars:
        preview = preview[: preview_chars - 3].rstrip() + "..."

    lines = [
        f"Rank: {hit.rank}",
        f"Score: {hit.score:.4f}",
        f"Document: {hit.document_id}",
        f"Clause: {hit.parent_clause}",
        f"Unit ID: {hit.unit_id}",
        f"Text preview: {preview}",
        "",
    ]
    return "\n".join(lines)


def validate_hits(hits: list[EvidenceUnit], *, known_unit_ids: set[str] | None = None) -> list[str]:
    """Return validation error messages; empty list means all checks passed."""
    errors: list[str] = []
    for hit in hits:
        if not hit.unit_id:
            errors.append(f"rank {hit.rank}: missing unit_id")
        elif known_unit_ids is not None and hit.unit_id not in known_unit_ids:
            errors.append(f"rank {hit.rank}: unknown unit_id {hit.unit_id}")
        if not hit.text or not hit.text.strip():
            errors.append(f"rank {hit.rank}: empty text for {hit.unit_id}")
        if hit.score is None:
            errors.append(f"rank {hit.rank}: missing score")
    return errors


def load_sample_questions(qa_file: Path, num_queries: int) -> list[str]:
    questions: list[str] = []
    with qa_file.open(encoding="utf-8") as fh:
        for line in fh:
            if len(questions) >= num_queries:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = (record.get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def run_single_query(
    retriever: DenseRetriever,
    query: str,
    *,
    top_k: int,
    preview_chars: int,
    known_unit_ids: set[str] | None = None,
) -> bool:
    print("=" * 72)
    print(f"Query: {query}")
    print("-" * 72)

    try:
        hits = retriever.retrieve(query, top_k)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    errors = validate_hits(hits, known_unit_ids=known_unit_ids)
    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    for hit in hits:
        print(format_hit(hit, preview_chars=preview_chars))

    print(f"Retrieved {len(hits)} evidence unit(s).")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.query and not args.qa_file:
        logging.error("Provide --query or --qa-file")
        return 1

    try:
        retriever = DenseRetriever(
            args.index,
            model_name=args.model,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1

    known_unit_ids = retriever.known_unit_ids

    ok = True
    if args.query:
        ok = run_single_query(
            retriever,
            args.query,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
            known_unit_ids=known_unit_ids,
        ) and ok

    if args.qa_file:
        qa_path = Path(args.qa_file)
        if not qa_path.is_file():
            logging.error("QA file not found: %s", qa_path)
            return 1
        questions = load_sample_questions(qa_path, args.num_queries)
        if not questions:
            logging.error("No questions loaded from %s", qa_path)
            return 1
        print(f"\nRunning {len(questions)} sample queries from {qa_path}\n")
        for question in questions:
            passed = run_single_query(
                retriever,
                question,
                top_k=args.top_k,
                preview_chars=args.preview_chars,
                known_unit_ids=known_unit_ids,
            )
            ok = passed and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
