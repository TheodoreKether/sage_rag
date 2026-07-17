"""Compare BM25 vs Dense retrieval results and generate case-study analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def load_metrics_json(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    return {key: float(value) for key, value in metrics.items()}


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


def _pct_delta(new: float, old: float) -> str:
    if old == 0:
        return "N/A" if new == 0 else "+inf"
    rel = (new - old) / old * 100
    sign = "+" if rel >= 0 else ""
    return f"{sign}{rel:.1f}%"


def _abs_delta(new: float, old: float) -> str:
    sign = "+" if new - old >= 0 else ""
    return f"{sign}{new - old:.4f}"


def _winner(bm25: float, dense: float) -> str:
    if abs(bm25 - dense) < 1e-9:
        return "tie"
    return "BM25" if bm25 > dense else "Dense"


def _recall_at_k(record: dict[str, Any], k: int) -> float:
    retrieved = record.get("retrieved_unit_ids") or []
    gold = record.get("gold_unit_ids") or []
    return compute_retrieval_metrics(retrieved, gold, recall_ks=(k,), ndcg_ks=())[f"Recall@{k}"]


def _doc_id(unit_id: str) -> str:
    return unit_id.split("::")[0] if unit_id else ""


def build_comparison_table(
    bm25_metrics: dict[str, float],
    dense_metrics: dict[str, float],
) -> str:
    lines = [
        "# BM25 vs Dense Retrieval Comparison",
        "",
        "Both retrievers were evaluated on the same QA dataset with identical metrics.",
        "",
        "## Metric Comparison",
        "",
        "| Metric | BM25 | Dense | Abs Δ | Rel Δ | Winner |",
        "|--------|-----:|------:|------:|------:|--------|",
    ]

    for metric in METRIC_NAMES:
        bm25 = bm25_metrics.get(metric, 0.0)
        dense = dense_metrics.get(metric, 0.0)
        lines.append(
            f"| {metric} | {bm25:.4f} | {dense:.4f} | "
            f"{_abs_delta(bm25, dense)} | {_pct_delta(bm25, dense)} | {_winner(bm25, dense)} |"
        )

    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"- BM25 Recall@10: **{bm25_metrics.get('Recall@10', 0.0):.2%}** vs Dense **{dense_metrics.get('Recall@10', 0.0):.2%}**",
            f"- BM25 MRR: **{bm25_metrics.get('MRR', 0.0):.4f}** vs Dense **{dense_metrics.get('MRR', 0.0):.4f}**",
            "",
        ]
    )
    return "\n".join(lines)


def build_case_study(
    bm25_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
) -> str:
    bm25_by_id = {r["qa_id"]: r for r in bm25_records}
    dense_by_id = {r["qa_id"]: r for r in dense_records}
    common_ids = sorted(set(bm25_by_id) & set(dense_by_id))

    bm25_wins: list[dict[str, Any]] = []
    dense_wins: list[dict[str, Any]] = []
    both_fail: list[dict[str, Any]] = []

    bm25_top_docs: Counter[str] = Counter()
    dense_top_docs: Counter[str] = Counter()
    doc_recall_bm25: Counter[str] = Counter()
    doc_recall_dense: Counter[str] = Counter()
    doc_total: Counter[str] = Counter()

    for qa_id in common_ids:
        b = bm25_by_id[qa_id]
        d = dense_by_id[qa_id]
        b_r10 = _recall_at_k(b, 10)
        d_r10 = _recall_at_k(d, 10)

        if b.get("retrieved_unit_ids"):
            bm25_top_docs[_doc_id(b["retrieved_unit_ids"][0])] += 1
        if d.get("retrieved_unit_ids"):
            dense_top_docs[_doc_id(d["retrieved_unit_ids"][0])] += 1

        gold_doc = _doc_id((b.get("gold_unit_ids") or [""])[0])
        if gold_doc:
            doc_total[gold_doc] += 1
            if b_r10 > 0:
                doc_recall_bm25[gold_doc] += 1
            if d_r10 > 0:
                doc_recall_dense[gold_doc] += 1

        if b_r10 > d_r10:
            bm25_wins.append(b)
        elif d_r10 > b_r10:
            dense_wins.append(d)
        elif b_r10 == 0 and d_r10 == 0:
            both_fail.append(b)

    lines = [
        "## Case Study",
        "",
        "### 1. Questions where BM25 outperforms Dense",
        "",
    ]
    lines.extend(_format_examples(bm25_wins[:5], label="BM25 better"))
    lines.extend(
        [
            "",
            "### 2. Questions where Dense outperforms BM25",
            "",
        ]
    )
    lines.extend(_format_examples(dense_wins[:5], label="Dense better"))
    lines.extend(
        [
            "",
            "### 3. Typical failure cases (both miss Recall@10)",
            "",
        ]
    )
    lines.extend(_format_examples(both_fail[:5], label="both miss"))
    lines.extend(
        [
            "",
            "### 4. Most frequently retrieved documents (Top-1)",
            "",
            "#### BM25",
            "",
            "| document_id | count |",
            "|-------------|------:|",
        ]
    )
    for doc, count in bm25_top_docs.most_common(10):
        lines.append(f"| `{doc}` | {count} |")

    lines.extend(
        [
            "",
            "#### Dense",
            "",
            "| document_id | count |",
            "|-------------|------:|",
        ]
    )
    for doc, count in dense_top_docs.most_common(10):
        lines.append(f"| `{doc}` | {count} |")

    lines.extend(
        [
            "",
            "### 5. Document-level Recall@10",
            "",
            "| document_id | QA count | BM25 Recall@10 | Dense Recall@10 |",
            "|-------------|---------:|---------------:|------------------:|",
        ]
    )
    for doc, total in doc_total.most_common(15):
        bm25_rate = doc_recall_bm25.get(doc, 0) / total
        dense_rate = doc_recall_dense.get(doc, 0) / total
        lines.append(
            f"| `{doc}` | {total} | {bm25_rate:.2%} | {dense_rate:.2%} |"
        )

    lines.extend(
        [
            "",
            "### Summary Counts",
            "",
            f"- Compared QA pairs: {len(common_ids)}",
            f"- BM25 wins (Recall@10): {len(bm25_wins)}",
            f"- Dense wins (Recall@10): {len(dense_wins)}",
            f"- Both fail (Recall@10): {len(both_fail)}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_examples(records: list[dict[str, Any]], *, label: str) -> list[str]:
    if not records:
        return ["_None._", ""]
    lines: list[str] = []
    for record in records:
        gold = ", ".join(record.get("gold_unit_ids") or [])
        retrieved = ", ".join((record.get("retrieved_unit_ids") or [])[:3])
        lines.append(f"- **{record.get('qa_id', 'unknown')}** ({label})")
        lines.append(f"  - Q: {record.get('question', '')[:120]}")
        lines.append(f"  - Gold: `{gold}`")
        lines.append(f"  - Top-3 retrieved: `{retrieved}`")
    lines.append("")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BM25 and Dense retrieval results.")
    parser.add_argument(
        "--bm25-results",
        default="results/retrieval/bm25/retrieval_results.jsonl",
    )
    parser.add_argument(
        "--dense-results",
        default="results/retrieval/dense/retrieval_results_v2.jsonl",
    )
    parser.add_argument(
        "--bm25-metrics",
        default="results/retrieval/bm25/metrics.json",
    )
    parser.add_argument(
        "--dense-metrics-json",
        default="",
        help="Optional dense metrics JSON; otherwise computed from dense results JSONL",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval/bm25_vs_dense.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    bm25_records = load_jsonl(Path(args.bm25_results))
    dense_records = load_jsonl(Path(args.dense_results))
    if not bm25_records:
        print(f"BM25 results not found or empty: {args.bm25_results}", file=sys.stderr)
        return 1
    if not dense_records:
        print(f"Dense results not found or empty: {args.dense_results}", file=sys.stderr)
        return 1

    bm25_metrics = load_metrics_json(Path(args.bm25_metrics)) or summarize_results(bm25_records)

    dense_metrics_path = Path(args.dense_metrics_json) if args.dense_metrics_json else None
    if dense_metrics_path and dense_metrics_path.is_file():
        dense_metrics = load_metrics_json(dense_metrics_path)
    else:
        dense_metrics = summarize_results(dense_records)

    report = build_comparison_table(bm25_metrics, dense_metrics)
    report += "\n" + build_case_study(bm25_records, dense_records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
