"""Evaluate dense retrieval against the QA benchmark dataset."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from src.retrieval.config import MODEL_NAME
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
    from src.evaluation.report import EvaluationSummary, write_report
    from src.evaluation.retrieval_metrics import average_metrics, compute_retrieval_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.retrieval.config import MODEL_NAME
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.retriever_base import EvidenceUnit, RetrieverBase
    from src.evaluation.report import EvaluationSummary, write_report
    from src.evaluation.retrieval_metrics import average_metrics, compute_retrieval_metrics

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path("results/retrieval_results.jsonl")
DEFAULT_REPORT_PATH = Path("results/retrieval_dense_report.md")


@dataclass
class QASample:
    qa_id: str
    question: str
    gold_unit_ids: list[str]
    question_type: str = ""
    document_id: str = ""


@dataclass
class EvaluationResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    summary: EvaluationSummary | None = None


def load_qa_samples(qa_path: Path) -> tuple[list[QASample], int]:
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")

    samples: list[QASample] = []
    skipped = 0

    with qa_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                logger.warning("Skipping corrupt QA JSON at line %d: %s", line_no, exc)
                continue

            question = (record.get("question") or "").strip()
            if not question:
                skipped += 1
                logger.warning("Skipping empty question at line %d", line_no)
                continue

            gold_ids = _extract_gold_unit_ids(record.get("supporting_evidence"))
            if not gold_ids:
                skipped += 1
                logger.warning("Skipping missing supporting_evidence at line %d", line_no)
                continue

            samples.append(
                QASample(
                    qa_id=str(record.get("qa_id", f"line_{line_no}")),
                    question=question,
                    gold_unit_ids=gold_ids,
                    question_type=str(record.get("question_type", "")),
                    document_id=str(record.get("document_id", "")),
                )
            )

    return samples, skipped


def _extract_gold_unit_ids(supporting_evidence: Any) -> list[str]:
    if not supporting_evidence or not isinstance(supporting_evidence, list):
        return []
    ids: list[str] = []
    for item in supporting_evidence:
        if isinstance(item, dict):
            uid = (item.get("unit_id") or "").strip()
            if uid:
                ids.append(uid)
    return ids


def sample_qa_pairs(samples: list[QASample], n: int, *, seed: int = 42) -> list[QASample]:
    if n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    return [samples[i] for i in sorted(rng.sample(range(len(samples)), n))]


def hits_to_retrieved(hits: list[EvidenceUnit]) -> list[dict[str, Any]]:
    return [
        {
            "rank": hit.rank,
            "unit_id": hit.unit_id,
            "score": round(float(hit.score), 6) if hit.score is not None else None,
        }
        for hit in hits
    ]


def evaluate_single_query(
    retriever: RetrieverBase,
    sample: QASample,
    *,
    top_k: int,
) -> dict[str, Any]:
    hits = retriever.retrieve(sample.question, top_k)
    retrieved = hits_to_retrieved(hits)
    retrieved_ids = [item["unit_id"] for item in retrieved]

    metrics = compute_retrieval_metrics(
        retrieved_ids,
        sample.gold_unit_ids,
        recall_ks=(1, 5, 10),
        ndcg_ks=(5, 10),
    )

    record: dict[str, Any] = {
        "qa_id": sample.qa_id,
        "question": sample.question,
        "gold_unit_ids": sample.gold_unit_ids,
        "retrieved": retrieved,
        "retrieved_unit_ids": retrieved_ids,
        **metrics,
    }
    if sample.question_type:
        record["question_type"] = sample.question_type
    if sample.document_id:
        record["document_id"] = sample.document_id
    return record


def compute_dataset_stats(samples: list[QASample]) -> dict[str, Any]:
    qtypes = Counter(s.question_type for s in samples if s.question_type)
    docs = Counter(s.document_id for s in samples if s.document_id)
    multi_gold = sum(1 for s in samples if len(s.gold_unit_ids) > 1)
    return {
        "unique question types": len(qtypes),
        "unique documents": len(docs),
        "multi-gold QA pairs": multi_gold,
    }


def _extract_metric_row(record: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(record[key])
        for key in record
        if key.startswith("Recall@") or key.startswith("nDCG@") or key == "MRR"
    }


def run_retrieval_evaluation(
    retriever: RetrieverBase,
    qa_path: Path,
    *,
    top_k: int = 10,
    sample_size: int | None = None,
    seed: int = 42,
    results_path: Path | None = None,
    report_path: Path | None = None,
    retriever_name: str = "DenseRetriever",
    model_name: str = "",
    index_dir: str = "",
) -> EvaluationResult:
    start = time.perf_counter()
    all_samples, load_skipped = load_qa_samples(qa_path)
    total_in_file = len(all_samples) + load_skipped

    eval_samples = sample_qa_pairs(all_samples, sample_size, seed=seed) if sample_size else all_samples

    records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    failed = 0

    for sample in tqdm(eval_samples, desc="Evaluating retrieval", unit="qa"):
        try:
            record = evaluate_single_query(retriever, sample, top_k=top_k)
            records.append(record)
            metric_rows.append(_extract_metric_row(record))
        except Exception as exc:
            failed += 1
            logger.warning("Retrieval failed for qa_id=%s: %s", sample.qa_id, exc)

    elapsed = time.perf_counter() - start
    summary = EvaluationSummary(
        retriever_name=retriever_name,
        model_name=model_name,
        index_dir=index_dir,
        qa_file=str(qa_path),
        top_k=top_k,
        total_qa_pairs=total_in_file,
        evaluated_pairs=len(records),
        skipped_pairs=load_skipped,
        failed_pairs=failed,
        elapsed_seconds=elapsed,
        average_metrics=average_metrics(metric_rows),
        dataset_stats=compute_dataset_stats(eval_samples),
    )

    if results_path:
        write_jsonl(records, results_path)
    if report_path:
        write_report(summary, report_path)

    return EvaluationResult(records=records, summary=summary)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d retrieval results to %s", len(records), path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Dense Retriever against QA ground-truth evidence.",
    )
    parser.add_argument(
        "--qa",
        default="data/qa_dataset/qa_pairs.jsonl",
        help="QA dataset JSONL path",
    )
    parser.add_argument(
        "--index",
        default="data/vector_store",
        help="Vector store directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of evidence units to retrieve per question",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate only a random subset of QA pairs (for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
        help="Per-question results JSONL path",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Markdown summary report path",
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
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.top_k <= 0:
        logger.error("--top-k must be positive")
        return 1

    try:
        retriever = DenseRetriever(args.index, model_name=args.model, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = run_retrieval_evaluation(
            retriever,
            Path(args.qa),
            top_k=args.top_k,
            sample_size=args.sample,
            seed=args.seed,
            results_path=Path(args.output),
            report_path=Path(args.report),
            retriever_name="DenseRetriever",
            model_name=args.model,
            index_dir=str(args.index),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    summary = result.summary
    if summary:
        logger.info(
            "Done: Recall@1=%.4f Recall@5=%.4f Recall@10=%.4f MRR=%.4f (n=%d, %.1fs)",
            summary.metric("Recall@1"),
            summary.metric("Recall@5"),
            summary.metric("Recall@10"),
            summary.metric("MRR"),
            summary.evaluated_pairs,
            summary.elapsed_seconds,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
