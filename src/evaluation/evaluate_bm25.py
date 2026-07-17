"""Evaluate BM25 retrieval against the QA benchmark dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from src.evaluation.evaluate_dense import run_retrieval_evaluation
    from src.evaluation.report import EvaluationSummary
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.config import DEFAULT_BM25_INDEX_DIR, DEFAULT_BM25_RESULTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate_dense import run_retrieval_evaluation
    from src.evaluation.report import EvaluationSummary
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.config import DEFAULT_BM25_INDEX_DIR, DEFAULT_BM25_RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path(DEFAULT_BM25_RESULTS_DIR) / "retrieval_results.jsonl"
DEFAULT_REPORT_PATH = Path(DEFAULT_BM25_RESULTS_DIR) / "evaluation_report.md"
DEFAULT_METRICS_PATH = Path(DEFAULT_BM25_RESULTS_DIR) / "metrics.json"


def render_bm25_report(summary: EvaluationSummary) -> str:
    lines = [
        "# BM25 Retrieval Evaluation Report",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| QA file | `{summary.qa_file}` |",
        f"| Total QA pairs in file | {summary.total_qa_pairs} |",
        f"| QA pairs evaluated | {summary.evaluated_pairs} |",
        f"| Skipped (invalid / empty) | {summary.skipped_pairs} |",
        f"| Failed (retrieval error) | {summary.failed_pairs} |",
    ]

    if summary.dataset_stats:
        for key, value in sorted(summary.dataset_stats.items()):
            lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Retriever Configuration",
            "",
            "| Setting | Value |",
            "|---------|-------|",
            f"| Retriever | `{summary.retriever_name}` |",
            f"| Index directory | `{summary.index_dir}` |",
            f"| top_k | {summary.top_k} |",
            "",
            "## Average Metrics",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Recall@1 | {summary.metric('Recall@1'):.4f} |",
            f"| Recall@5 | {summary.metric('Recall@5'):.4f} |",
            f"| Recall@10 | {summary.metric('Recall@10'):.4f} |",
            f"| MRR | {summary.metric('MRR'):.4f} |",
            f"| nDCG@5 | {summary.metric('nDCG@5'):.4f} |",
            f"| nDCG@10 | {summary.metric('nDCG@10'):.4f} |",
            "",
            "## Runtime",
            "",
            f"- Total evaluation time: **{summary.elapsed_seconds:.2f} s**",
            "",
            "## Notes",
            "",
            "- Ground truth: `supporting_evidence[].unit_id` from the QA dataset.",
            "- Per-question results are saved in `results/retrieval/bm25/retrieval_results.jsonl`.",
            "- Metrics are computed by the shared evaluation pipeline used for Dense retrieval.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bm25_report(summary: EvaluationSummary, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_bm25_report(summary) + "\n", encoding="utf-8")
    logger.info("Wrote BM25 evaluation report to %s", report_path)


def write_metrics_json(summary: EvaluationSummary, metrics_path: Path) -> None:
    payload = {
        "retriever": summary.retriever_name,
        "qa_file": summary.qa_file,
        "index_dir": summary.index_dir,
        "top_k": summary.top_k,
        "evaluated_pairs": summary.evaluated_pairs,
        "elapsed_seconds": summary.elapsed_seconds,
        "metrics": summary.average_metrics,
        "dataset_stats": summary.dataset_stats,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote metrics to %s", metrics_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 Retriever against QA ground-truth evidence.",
    )
    parser.add_argument(
        "--qa",
        default="data/qa_dataset/qa_pairs_v2.jsonl",
        help="QA dataset JSONL path",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_BM25_INDEX_DIR,
        help="BM25 index directory",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH))
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
        retriever = BM25Retriever(args.index)
    except (FileNotFoundError, ValueError) as exc:
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
            report_path=None,
            retriever_name="BM25Retriever",
            model_name="BM25Okapi",
            index_dir=str(args.index),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    summary = result.summary
    if summary:
        write_bm25_report(summary, Path(args.report))
        write_metrics_json(summary, Path(args.metrics))
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
