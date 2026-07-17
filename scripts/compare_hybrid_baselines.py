"""Compare Hybrid (RRF) vs Dense and BM25 retrieval baselines."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import average_metrics, compute_retrieval_metrics

METRIC_NAMES = [
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "MRR",
    "nDCG@5",
    "nDCG@10",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(record: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(record[key])
        for key in record
        if key.startswith("Recall@") or key.startswith("nDCG@") or key == "MRR"
    }


def summarize_results(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    return average_metrics([_metric_row(r) for r in records])


def avg_latency_ms(metrics_payload: dict[str, Any], records: list[dict[str, Any]]) -> float:
    if "avg_query_latency_ms" in metrics_payload:
        return float(metrics_payload["avg_query_latency_ms"])
    latencies = [float(r["query_latency_ms"]) for r in records if "query_latency_ms" in r]
    if latencies:
        return sum(latencies) / len(latencies)
    elapsed = float(metrics_payload.get("elapsed_seconds", 0.0))
    n = int(metrics_payload.get("evaluated_pairs", len(records)))
    return (elapsed / n * 1000.0) if n else 0.0


def _recall_at_k(record: dict[str, Any], k: int) -> float:
    retrieved = record.get("retrieved_unit_ids") or []
    gold = record.get("gold_unit_ids") or []
    return compute_retrieval_metrics(retrieved, gold, recall_ks=(k,), ndcg_ks=())[f"Recall@{k}"]


def _doc_id(unit_id: str) -> str:
    return unit_id.split("::")[0] if unit_id else ""


def _doc_family(doc_id: str) -> str:
    return re.sub(r"-\d{4}$", "", doc_id)


def _clause_prefix(unit_id: str) -> str:
    parts = unit_id.split("::")
    if len(parts) < 3:
        return unit_id
    return "::".join(parts[:-1])


def classify_failure(gold_ids: list[str], retrieved_ids: list[str]) -> str:
    if not gold_ids or not retrieved_ids:
        return "no_retrieval"
    gold = gold_ids[0]
    top = retrieved_ids[0]
    if top in gold_ids:
        return "success"

    g_doc = _doc_id(gold)
    r_doc = _doc_id(top)
    if g_doc != r_doc:
        if _doc_family(g_doc) == _doc_family(r_doc):
            return "standard version confusion"
        return "cross-document confusion"

    if _clause_prefix(gold) == _clause_prefix(top):
        return "parent-child clause confusion"

    g_tokens = set(re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", gold.lower()))
    q_overlap = len(g_tokens & set(re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", top.lower())))
    if q_overlap == 0:
        return "synonym / semantic mismatch"

    return "same-document clause mismatch"


def build_comparison_table(
    bm25: dict[str, float],
    dense: dict[str, float],
    hybrid: dict[str, float],
    *,
    bm25_latency: float,
    dense_latency: float,
    hybrid_latency: float,
) -> str:
    lines = [
        "# Hybrid vs Baselines Retrieval Comparison",
        "",
        "Hybrid = Dense Top-100 + BM25 Top-100 fused with RRF (k=60).",
        "",
        "## Metric Comparison",
        "",
        "| Metric | BM25 | Dense | Hybrid (RRF) | Best |",
        "|--------|-----:|------:|-------------:|------|",
    ]

    best_map: dict[str, str] = {}
    for metric in METRIC_NAMES:
        values = {"BM25": bm25.get(metric, 0.0), "Dense": dense.get(metric, 0.0), "Hybrid": hybrid.get(metric, 0.0)}
        best = max(values, key=values.get)
        best_map[metric] = best
        lines.append(
            f"| {metric} | {values['BM25']:.4f} | {values['Dense']:.4f} | "
            f"{values['Hybrid']:.4f} | {best} |"
        )

    lines.extend(
        [
            f"| Avg Query Time (ms) | {bm25_latency:.2f} | {dense_latency:.2f} | {hybrid_latency:.2f} | — |",
            "",
            "## Key Finding",
            "",
            f"- Hybrid Recall@10: **{hybrid.get('Recall@10', 0.0):.2%}** "
            f"(BM25 {bm25.get('Recall@10', 0.0):.2%}, Dense {dense.get('Recall@10', 0.0):.2%})",
            f"- Hybrid MRR: **{hybrid.get('MRR', 0.0):.4f}** "
            f"(BM25 {bm25.get('MRR', 0.0):.4f}, Dense {dense.get('MRR', 0.0):.4f})",
            f"- Hybrid avg latency: **{hybrid_latency:.2f} ms/query**",
            "",
        ]
    )
    return "\n".join(lines)


def build_case_study(
    hybrid_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    bm25_records: list[dict[str, Any]],
) -> str:
    hybrid_by_id = {r["qa_id"]: r for r in hybrid_records}
    dense_by_id = {r["qa_id"]: r for r in dense_records}
    bm25_by_id = {r["qa_id"]: r for r in bm25_records}
    common_ids = sorted(set(hybrid_by_id) & set(dense_by_id) & set(bm25_by_id))

    improved_over_dense: list[dict[str, Any]] = []
    improved_over_bm25: list[dict[str, Any]] = []
    hybrid_failures: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()

    for qa_id in common_ids:
        h = hybrid_by_id[qa_id]
        d = dense_by_id[qa_id]
        b = bm25_by_id[qa_id]
        h_r10 = _recall_at_k(h, 10)
        d_r10 = _recall_at_k(d, 10)
        b_r10 = _recall_at_k(b, 10)

        if h_r10 > d_r10:
            improved_over_dense.append(h)
        if h_r10 > b_r10:
            improved_over_bm25.append(h)
        if h_r10 == 0:
            reason = classify_failure(h.get("gold_unit_ids") or [], h.get("retrieved_unit_ids") or [])
            failure_reasons[reason] += 1
            hybrid_failures.append({**h, "failure_reason": reason})

    lines = [
        "## Case Study",
        "",
        "### 1. Queries improved over Dense",
        "",
    ]
    lines.extend(_format_examples(improved_over_dense[:5]))
    lines.extend(["", "### 2. Queries improved over BM25", ""])
    lines.extend(_format_examples(improved_over_bm25[:5]))
    lines.extend(["", "### 3. Queries where Hybrid still fails (Recall@10 = 0)", ""])
    lines.extend(_format_examples(hybrid_failures[:5], include_reason=True))
    lines.extend(
        [
            "",
            "### 4. Failure reason distribution (Hybrid misses)",
            "",
            "| Reason | Count |",
            "|--------|------:|",
        ]
    )
    for reason, count in failure_reasons.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "### Summary Counts",
            "",
            f"- Compared QA pairs: {len(common_ids)}",
            f"- Hybrid better than Dense (Recall@10): {len(improved_over_dense)}",
            f"- Hybrid better than BM25 (Recall@10): {len(improved_over_bm25)}",
            f"- Hybrid failures (Recall@10 = 0): {len(hybrid_failures)}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_examples(
    records: list[dict[str, Any]],
    *,
    include_reason: bool = False,
) -> list[str]:
    if not records:
        return ["_None._", ""]
    lines: list[str] = []
    for record in records:
        gold = ", ".join(record.get("gold_unit_ids") or [])
        retrieved = ", ".join((record.get("retrieved_unit_ids") or [])[:3])
        lines.append(f"- **{record.get('qa_id', 'unknown')}**")
        lines.append(f"  - Q: {record.get('question', '')[:120]}")
        lines.append(f"  - Gold: `{gold}`")
        lines.append(f"  - Top-3 retrieved: `{retrieved}`")
        if include_reason:
            lines.append(f"  - Failure reason: {record.get('failure_reason', 'unknown')}")
    lines.append("")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Hybrid vs BM25 and Dense.")
    parser.add_argument("--bm25-results", default="results/retrieval/bm25/retrieval_results.jsonl")
    parser.add_argument("--dense-results", default="results/retrieval/dense/retrieval_results_v2.jsonl")
    parser.add_argument("--hybrid-results", default="results/retrieval/hybrid/retrieval_results.jsonl")
    parser.add_argument("--bm25-metrics", default="results/retrieval/bm25/metrics.json")
    parser.add_argument("--dense-metrics", default="results/retrieval/dense/metrics_v2.json")
    parser.add_argument("--hybrid-metrics", default="results/retrieval/hybrid/metrics.json")
    parser.add_argument("--output", default="results/retrieval/hybrid_vs_baselines.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    bm25_records = load_jsonl(Path(args.bm25_results))
    dense_records = load_jsonl(Path(args.dense_results))
    hybrid_records = load_jsonl(Path(args.hybrid_results))
    if not hybrid_records:
        print(f"Hybrid results not found: {args.hybrid_results}", file=sys.stderr)
        return 1

    bm25_payload = load_metrics(Path(args.bm25_metrics))
    dense_payload = load_metrics(Path(args.dense_metrics))
    hybrid_payload = load_metrics(Path(args.hybrid_metrics))

    bm25_metrics = bm25_payload.get("metrics") or summarize_results(bm25_records)
    dense_metrics = dense_payload.get("metrics") or summarize_results(dense_records)
    hybrid_metrics = hybrid_payload.get("metrics") or summarize_results(hybrid_records)

    report = build_comparison_table(
        bm25_metrics,
        dense_metrics,
        hybrid_metrics,
        bm25_latency=avg_latency_ms(bm25_payload, bm25_records),
        dense_latency=avg_latency_ms(dense_payload, dense_records),
        hybrid_latency=avg_latency_ms(hybrid_payload, hybrid_records),
    )
    if bm25_records and dense_records:
        report += "\n" + build_case_study(hybrid_records, dense_records, bm25_records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
