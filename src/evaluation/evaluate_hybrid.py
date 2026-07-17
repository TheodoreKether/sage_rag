"""Evaluate Hybrid (RRF) retrieval against the QA benchmark dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

try:
    from src.evaluation.evaluate_dense import (
        compute_dataset_stats,
        evaluate_single_query,
        load_qa_samples,
        sample_qa_pairs,
        write_jsonl,
        _extract_metric_row,
    )
    from src.evaluation.report import EvaluationSummary
    from src.evaluation.retrieval_metrics import average_metrics
    from src.retrieval.config import (
        DEFAULT_BM25_INDEX_DIR,
        DEFAULT_FUSION_TOP_N,
        DEFAULT_HYBRID_RESULTS_DIR,
        DEFAULT_INDEX_DIR,
        DEFAULT_RRF_K,
        MODEL_NAME,
    )
    from src.retrieval.hybrid import HybridRetriever
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate_dense import (
        compute_dataset_stats,
        evaluate_single_query,
        load_qa_samples,
        sample_qa_pairs,
        write_jsonl,
        _extract_metric_row,
    )
    from src.evaluation.report import EvaluationSummary
    from src.evaluation.retrieval_metrics import average_metrics
    from src.retrieval.config import (
        DEFAULT_BM25_INDEX_DIR,
        DEFAULT_FUSION_TOP_N,
        DEFAULT_HYBRID_RESULTS_DIR,
        DEFAULT_INDEX_DIR,
        DEFAULT_RRF_K,
        MODEL_NAME,
    )
    from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path(DEFAULT_HYBRID_RESULTS_DIR) / "retrieval_results.jsonl"
DEFAULT_REPORT_PATH = Path(DEFAULT_HYBRID_RESULTS_DIR) / "evaluation_report.md"
DEFAULT_METRICS_PATH = Path(DEFAULT_HYBRID_RESULTS_DIR) / "metrics.json"


def run_hybrid_evaluation(
    retriever: HybridRetriever,
    qa_path: Path,
    *,
    top_k: int = 10,
    sample_size: int | None = None,
    seed: int = 42,
    results_path: Path | None = None,
) -> tuple[list[dict], EvaluationSummary]:
    start = time.perf_counter()
    all_samples, load_skipped = load_qa_samples(qa_path)
    total_in_file = len(all_samples) + load_skipped
    eval_samples = sample_qa_pairs(all_samples, sample_size, seed=seed) if sample_size else all_samples

    records: list[dict] = []
    metric_rows: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    failed = 0

    for sample in tqdm(eval_samples, desc="Evaluating hybrid retrieval", unit="qa"):
        try:
            query_start = time.perf_counter()
            record = evaluate_single_query(retriever, sample, top_k=top_k)
            latency_ms = (time.perf_counter() - query_start) * 1000.0
            record["query_latency_ms"] = round(latency_ms, 3)
            latencies_ms.append(latency_ms)
            records.append(record)
            metric_rows.append(_extract_metric_row(record))
        except Exception as exc:
            failed += 1
            logger.warning("Retrieval failed for qa_id=%s: %s", sample.qa_id, exc)

    elapsed = time.perf_counter() - start
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    summary = EvaluationSummary(
        retriever_name="HybridRetriever (RRF)",
        model_name=MODEL_NAME,
        index_dir=f"dense={retriever.dense_index_dir}, bm25={retriever.bm25_index_dir}",
        qa_file=str(qa_path),
        top_k=top_k,
        total_qa_pairs=total_in_file,
        evaluated_pairs=len(records),
        skipped_pairs=load_skipped,
        failed_pairs=failed,
        elapsed_seconds=elapsed,
        average_metrics=average_metrics(metric_rows),
        dataset_stats={
            **compute_dataset_stats(eval_samples),
            "avg_query_latency_ms": round(avg_latency_ms, 3),
            "rrf_k": retriever.rrf_k,
            "fusion_top_n": retriever.fusion_top_n,
        },
    )

    if results_path:
        write_jsonl(records, results_path)

    return records, summary


def render_hybrid_report(summary: EvaluationSummary) -> str:
    avg_latency = summary.dataset_stats.get("avg_query_latency_ms", 0.0)
    rrf_k = summary.dataset_stats.get("rrf_k", DEFAULT_RRF_K)
    fusion_top_n = summary.dataset_stats.get("fusion_top_n", DEFAULT_FUSION_TOP_N)

    lines = [
        "# Hybrid Retrieval Evaluation Report (RRF)",
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

    for key, value in sorted(summary.dataset_stats.items()):
        if key not in {"rrf_k", "fusion_top_n", "avg_query_latency_ms"}:
            lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Retriever Configuration",
            "",
            "| Setting | Value |",
            "|---------|-------|",
            f"| Retriever | `{summary.retriever_name}` |",
            f"| Dense index | `{summary.index_dir}` |",
            f"| Fusion algorithm | Reciprocal Rank Fusion |",
            f"| RRF k | {rrf_k} |",
            f"| Candidate pool per retriever | Top-{fusion_top_n} |",
            f"| Final top_k | {summary.top_k} |",
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
            f"- Average query latency: **{avg_latency:.2f} ms**",
            "",
            "## Notes",
            "",
            "- Dense Top-100 and BM25 Top-100 are fused with RRF; raw scores are not used.",
            "- Per-question results are saved in `results/retrieval/hybrid/retrieval_results.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_metrics_json(summary: EvaluationSummary, metrics_path: Path) -> None:
    payload = {
        "retriever": summary.retriever_name,
        "qa_file": summary.qa_file,
        "index_dir": summary.index_dir,
        "top_k": summary.top_k,
        "evaluated_pairs": summary.evaluated_pairs,
        "elapsed_seconds": summary.elapsed_seconds,
        "avg_query_latency_ms": summary.dataset_stats.get("avg_query_latency_ms", 0.0),
        "rrf_k": summary.dataset_stats.get("rrf_k", DEFAULT_RRF_K),
        "fusion_top_n": summary.dataset_stats.get("fusion_top_n", DEFAULT_FUSION_TOP_N),
        "metrics": summary.average_metrics,
        "dataset_stats": summary.dataset_stats,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote metrics to %s", metrics_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid (RRF) Retriever against QA ground-truth evidence.",
    )
    parser.add_argument("--qa", default="data/qa_dataset/qa_pairs_v2.jsonl")
    parser.add_argument("--dense-index", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--bm25-index", default=DEFAULT_BM25_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--fusion-top-n", type=int, default=DEFAULT_FUSION_TOP_N)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default="auto")
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
        retriever = HybridRetriever(
            args.dense_index,
            args.bm25_index,
            model_name=args.model,
            device=args.device,
            rrf_k=args.rrf_k,
            fusion_top_n=args.fusion_top_n,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        _, summary = run_hybrid_evaluation(
            retriever,
            Path(args.qa),
            top_k=args.top_k,
            sample_size=args.sample,
            seed=args.seed,
            results_path=Path(args.output),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    write_metrics_json(summary, Path(args.metrics))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_hybrid_report(summary) + "\n", encoding="utf-8")
    logger.info("Wrote hybrid evaluation report to %s", report_path)

    logger.info(
        "Done: Recall@1=%.4f Recall@5=%.4f Recall@10=%.4f MRR=%.4f "
        "avg_latency=%.2fms (n=%d, %.1fs)",
        summary.metric("Recall@1"),
        summary.metric("Recall@5"),
        summary.metric("Recall@10"),
        summary.metric("MRR"),
        summary.dataset_stats.get("avg_query_latency_ms", 0.0),
        summary.evaluated_pairs,
        summary.elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
