"""Evaluate Dense/BM25/Hybrid on clean QA and write comparison reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_dense import run_retrieval_evaluation
from src.evaluation.report import write_report
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

QA_CLEAN = ROOT / "data/qa_dataset/qa_pairs_clean.jsonl"
QA_V2 = ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
OUT_DIR = ROOT / "results/retrieval/clean_benchmark"

ORIGINAL_METRICS = {
    "bm25": {"Recall@10": 0.7866, "MRR": 0.6071, "Recall@1": 0.5081, "Recall@5": 0.7276},
    "dense": {"Recall@10": 0.6199, "MRR": 0.4416, "Recall@1": 0.3659, "Recall@5": 0.5366},
    "hybrid": {"Recall@10": 0.7419, "MRR": 0.5447, "Recall@1": 0.4492, "Recall@5": 0.6789},
}


def _summary_to_metrics(summary: Any) -> dict[str, float]:
    keys = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]
    out: dict[str, float] = {}
    for k in keys:
        try:
            out[k] = float(summary.metric(k))
        except Exception:
            continue
    return out


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_all(qa_path: Path, out_dir: Path, top_k: int = 10) -> dict[str, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}

    print("Evaluating Dense...")
    dense = DenseRetriever("data/vector_store")
    dense_result = run_retrieval_evaluation(
        dense,
        qa_path,
        top_k=top_k,
        results_path=out_dir / "dense_results.jsonl",
        report_path=None,
        retriever_name="DenseRetriever",
        model_name="BAAI/bge-m3",
        index_dir="data/vector_store",
    )
    assert dense_result.summary is not None
    dense_m = _summary_to_metrics(dense_result.summary)
    metrics["dense"] = dense_m
    write_metrics_json(
        out_dir / "dense_metrics.json",
        {
            "retriever": "DenseRetriever",
            "qa_file": str(qa_path),
            "evaluated_pairs": dense_result.summary.evaluated_pairs,
            "metrics": dense_m,
        },
    )
    write_report(dense_result.summary, out_dir / "dense_report.md")

    print("Evaluating BM25...")
    bm25 = BM25Retriever("data/bm25_index")
    bm25_result = run_retrieval_evaluation(
        bm25,
        qa_path,
        top_k=top_k,
        results_path=out_dir / "bm25_results.jsonl",
        report_path=None,
        retriever_name="BM25Retriever",
        model_name="BM25Okapi",
        index_dir="data/bm25_index",
    )
    assert bm25_result.summary is not None
    bm25_m = _summary_to_metrics(bm25_result.summary)
    metrics["bm25"] = bm25_m
    write_metrics_json(
        out_dir / "bm25_metrics.json",
        {
            "retriever": "BM25Retriever",
            "qa_file": str(qa_path),
            "evaluated_pairs": bm25_result.summary.evaluated_pairs,
            "metrics": bm25_m,
        },
    )
    write_report(bm25_result.summary, out_dir / "bm25_report.md")

    print("Evaluating Hybrid...")
    hybrid = HybridRetriever(
        dense_index_dir="data/vector_store",
        bm25_index_dir="data/bm25_index",
    )
    hybrid_result = run_retrieval_evaluation(
        hybrid,
        qa_path,
        top_k=top_k,
        results_path=out_dir / "hybrid_results.jsonl",
        report_path=None,
        retriever_name="HybridRetriever (RRF)",
        model_name="Dense+BM25+RRF",
        index_dir="dense=data/vector_store, bm25=data/bm25_index",
    )
    assert hybrid_result.summary is not None
    hybrid_m = _summary_to_metrics(hybrid_result.summary)
    metrics["hybrid"] = hybrid_m
    write_metrics_json(
        out_dir / "hybrid_metrics.json",
        {
            "retriever": "HybridRetriever (RRF)",
            "qa_file": str(qa_path),
            "evaluated_pairs": hybrid_result.summary.evaluated_pairs,
            "metrics": hybrid_m,
        },
    )
    write_report(hybrid_result.summary, out_dir / "hybrid_report.md")

    return metrics


def write_evaluation_report(
    metrics: dict[str, dict[str, float]],
    *,
    clean_n: int,
    original_n: int,
    removed: int,
    path: Path,
) -> None:
    lines = [
        "# Clean Benchmark Retrieval Evaluation",
        "",
        f"- QA: `data/qa_dataset/qa_pairs_clean.jsonl` (**{clean_n}** queries)",
        f"- Removed Dataset Issues from V2: **{removed}** (from {original_n})",
        "- Retrievers unchanged (Dense / BM25 / Hybrid RRF)",
        "",
        "## Metrics (Clean)",
        "",
        "| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|-----------|---------:|---------:|----------:|----:|--------:|",
    ]
    for name in ("bm25", "dense", "hybrid"):
        m = metrics[name]
        lines.append(
            f"| {name.upper()} | {m.get('Recall@1', 0):.4f} | {m.get('Recall@5', 0):.4f} | "
            f"{m.get('Recall@10', 0):.4f} | {m.get('MRR', 0):.4f} | {m.get('nDCG@10', 0):.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Per-query results: `*_results.jsonl`",
        "- Machine-readable metrics: `*_metrics.json`",
        "- Comparison vs original V2: `comparison.md`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(
    metrics: dict[str, dict[str, float]],
    *,
    clean_n: int,
    original_n: int,
    removed: int,
    path: Path,
) -> None:
    lines = [
        "# Benchmark Quality Impact",
        "",
        "Effect of removing Dataset / Annotation Issues from QA V2.",
        "",
        f"- Original QA V2: **{original_n}**",
        f"- Clean QA: **{clean_n}** (removed **{removed}**)",
        "",
        "## Recall@10",
        "",
        "| Method | Original Recall@10 | Clean Recall@10 | Δ |",
        "|--------|-------------------:|----------------:|--:|",
    ]
    for name, label in (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid")):
        orig = ORIGINAL_METRICS[name]["Recall@10"]
        clean = metrics[name].get("Recall@10", 0.0)
        delta = clean - orig
        lines.append(
            f"| {label} | {orig * 100:.2f}% | {clean * 100:.2f}% | {delta * 100:+.2f} pp |"
        )

    lines += [
        "",
        "## MRR",
        "",
        "| Method | Original MRR | Clean MRR | Δ |",
        "|--------|-------------:|----------:|--:|",
    ]
    for name, label in (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid")):
        orig = ORIGINAL_METRICS[name]["MRR"]
        clean = metrics[name].get("MRR", 0.0)
        delta = clean - orig
        lines.append(f"| {label} | {orig:.4f} | {clean:.4f} | {delta:+.4f} |")

    lines += [
        "",
        "## Analysis",
        "",
        "- Cleaning removes question–gold mismatches that inflate apparent retrieval failures.",
        "- If Clean Recall@10 rises, part of the previous error mass was **annotation noise**, "
        "not retriever limitation.",
        "- Residual misses on the clean set are the fair basis for SAGE-RAG motivation "
        "(version / appendix / cross-reference / hierarchy / table / semantic).",
        "- Relative ranking among BM25 / Dense / Hybrid should be interpreted on the clean set.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-benchmark retrieval evaluation")
    parser.add_argument("--qa", type=Path, default=QA_CLEAN)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-eval", action="store_true", help="Only write comparison from existing metrics")
    args = parser.parse_args()

    cleaning = {}
    log_path = args.out_dir / "cleaning_log.json"
    if log_path.is_file():
        cleaning = json.loads(log_path.read_text(encoding="utf-8"))
    original_n = int(cleaning.get("original_count") or len(list(QA_V2.open(encoding="utf-8"))))
    clean_n = int(cleaning.get("clean_count") or sum(1 for _ in args.qa.open(encoding="utf-8") if _.strip()))
    removed = int(cleaning.get("removed_count") or (original_n - clean_n))

    if args.skip_eval:
        metrics = {
            name: json.loads((args.out_dir / f"{name}_metrics.json").read_text(encoding="utf-8"))[
                "metrics"
            ]
            for name in ("bm25", "dense", "hybrid")
        }
    else:
        metrics = run_all(args.qa, args.out_dir, top_k=args.top_k)

    write_evaluation_report(
        metrics,
        clean_n=clean_n,
        original_n=original_n,
        removed=removed,
        path=args.out_dir / "evaluation_report.md",
    )
    write_comparison(
        metrics,
        clean_n=clean_n,
        original_n=original_n,
        removed=removed,
        path=args.out_dir / "comparison.md",
    )
    print("Clean evaluation complete.")
    for name in ("bm25", "dense", "hybrid"):
        print(f"  {name}: Recall@10={metrics[name].get('Recall@10', 0):.4f}")


if __name__ == "__main__":
    main()
