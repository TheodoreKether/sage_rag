"""Compare V1 vs V2 QA benchmark datasets and Dense retrieval results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval_metrics import average_metrics


def load_qa_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_retrieval_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def qa_dataset_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    qtypes = Counter(r.get("question_type", "unknown") for r in records)
    q_lens = [len((r.get("question") or "")) for r in records]
    a_lens = [len((r.get("answer") or "")) for r in records]
    docs = Counter(r.get("document_id", "") for r in records)
    return {
        "count": len(records),
        "question_type_distribution": dict(sorted(qtypes.items())),
        "avg_question_length": sum(q_lens) / len(q_lens) if q_lens else 0.0,
        "avg_answer_length": sum(a_lens) / len(a_lens) if a_lens else 0.0,
        "unique_documents": len(docs),
    }


def _metric_row(record: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(record[key])
        for key in record
        if key.startswith("Recall@") or key.startswith("nDCG@") or key == "MRR"
    }


def retrieval_summary(records: list[dict[str, Any]]) -> dict[str, float]:
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


def parse_unit_id(unit_id: str) -> dict[str, str]:
    parts = unit_id.split("::")
    doc = parts[0] if parts else ""
    clause_path = "::".join(parts[1:-1]) if len(parts) > 2 else ""
    parent = parts[-2] if len(parts) >= 2 else ""
    return {"document_id": doc, "clause_path": clause_path, "parent_clause": parent}


def _doc_family(doc_id: str) -> str:
    return re.sub(r"-\d{4}$", "", doc_id)


def _clause_prefix(unit_id: str) -> str:
    parts = unit_id.split("::")
    if len(parts) < 3:
        return unit_id
    return "::".join(parts[:-1])


def classify_failure(
    gold_ids: list[str],
    retrieved_ids: list[str],
    question: str,
) -> str:
    if not gold_ids or not retrieved_ids:
        return "no_retrieval"
    gold = gold_ids[0]
    top = retrieved_ids[0]
    if top in gold_ids:
        return "success"

    g = parse_unit_id(gold)
    r = parse_unit_id(top)

    if g["document_id"] != r["document_id"]:
        if _doc_family(g["document_id"]) == _doc_family(r["document_id"]):
            return "cross-version confusion"
        return "cross-document confusion"

    g_prefix = _clause_prefix(gold)
    r_prefix = _clause_prefix(top)
    if g_prefix == r_prefix:
        return "sibling clause confusion"

    if gold.startswith(r_prefix) or r_prefix.startswith(gold.split("::")[0]):
        return "parent-child clause confusion"
    if r_prefix.startswith(g_prefix) or gold.startswith(top):
        return "parent-child clause confusion"

    if len(question) < 25 or question.count("「") == 0 and len(question) < 40:
        return "overly generic question"

    return "semantic mismatch"


def analyze_failures(records: list[dict[str, Any]], *, top_n: int = 20) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()
    wrong_doc_counts: Counter[str] = Counter()

    for rec in records:
        gold_ids = rec.get("gold_unit_ids") or []
        retrieved_ids = rec.get("retrieved_unit_ids") or []
        hit = bool(set(gold_ids) & set(retrieved_ids[:10]))
        pattern = classify_failure(gold_ids, retrieved_ids, rec.get("question", ""))
        if not hit:
            mrr = float(rec.get("MRR", 0.0))
            failures.append(
                {
                    "qa_id": rec.get("qa_id", ""),
                    "question": rec.get("question", ""),
                    "gold_unit_ids": gold_ids,
                    "top_retrieved": retrieved_ids[:3],
                    "top_score": (rec.get("retrieved") or [{}])[0].get("score"),
                    "pattern": pattern,
                    "MRR": mrr,
                }
            )
            pattern_counts[pattern] += 1
            if retrieved_ids:
                wrong_doc = parse_unit_id(retrieved_ids[0])["document_id"]
                wrong_doc_counts[wrong_doc] += 1

    failures.sort(key=lambda x: (x["MRR"], x.get("top_score") or 0))
    return {
        "failure_count": len(failures),
        "failure_patterns": dict(pattern_counts.most_common()),
        "top_difficult": failures[:top_n],
        "top_wrong_documents": wrong_doc_counts.most_common(15),
    }


def retrieval_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    qtype_recall: dict[str, list[float]] = defaultdict(list)
    doc_recall: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    success_scores: list[float] = []
    fail_scores: list[float] = []
    retrieved_doc_counts: Counter[str] = Counter()

    for rec in records:
        qtype = rec.get("question_type", "unknown")
        doc = rec.get("document_id", "unknown")
        r10 = float(rec.get("Recall@10", 0.0))
        qtype_recall[qtype].append(r10)
        doc_recall[doc].append(r10)

        gold_ids = set(rec.get("gold_unit_ids") or [])
        retrieved = rec.get("retrieved") or []
        hit_at_10 = bool(set(rec.get("retrieved_unit_ids", [])[:10]) & gold_ids)

        for item in retrieved[:10]:
            score = item.get("score")
            if score is not None:
                all_scores.append(float(score))
            uid = item.get("unit_id", "")
            if uid:
                retrieved_doc_counts[parse_unit_id(uid)["document_id"]] += 1

        if retrieved:
            top_score = retrieved[0].get("score")
            if top_score is not None:
                if hit_at_10:
                    success_scores.append(float(top_score))
                else:
                    fail_scores.append(float(top_score))

    qtype_avg = {
        k: sum(v) / len(v) for k, v in sorted(qtype_recall.items()) if v
    }
    doc_avg = {
        k: sum(v) / len(v) for k, v in sorted(doc_recall.items()) if v
    }

    return {
        "question_type_recall_at_10": qtype_avg,
        "document_recall_at_10": doc_avg,
        "avg_retrieved_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "avg_success_top1_score": sum(success_scores) / len(success_scores) if success_scores else 0.0,
        "avg_failure_top1_score": sum(fail_scores) / len(fail_scores) if fail_scores else 0.0,
        "top_retrieved_documents": retrieved_doc_counts.most_common(10),
    }


def render_comparison_report(
    *,
    v1_qa_stats: dict[str, Any],
    v2_qa_stats: dict[str, Any],
    v1_metrics: dict[str, float],
    v2_metrics: dict[str, float],
    v2_failures: dict[str, Any],
    v1_path: str,
    v2_path: str,
) -> str:
    metric_names = ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]
    lines = [
        "# QA Benchmark V1 vs V2 — Dense Retrieval Comparison",
        "",
        "## Overview",
        "",
        "This report compares the legacy template-style QA benchmark (V1) against the",
        "natural-language retrieval benchmark (V2). The retriever, embedding model, and",
        "vector index are identical (BGE-M3 + FAISS IndexFlatIP).",
        "",
        "| Dataset | File | QA Pairs |",
        "|---------|------|--------:|",
        f"| V1 (template) | `{v1_path}` | {v1_qa_stats['count']} |",
        f"| V2 (natural) | `{v2_path}` | {v2_qa_stats['count']} |",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | V1 | V2 |",
        "|--------|---:|---:|",
        f"| Average question length (chars) | {v1_qa_stats['avg_question_length']:.1f} | {v2_qa_stats['avg_question_length']:.1f} |",
        f"| Average answer length (chars) | {v1_qa_stats['avg_answer_length']:.1f} | {v2_qa_stats['avg_answer_length']:.1f} |",
        f"| Unique documents | {v1_qa_stats['unique_documents']} | {v2_qa_stats['unique_documents']} |",
        "",
        "### V1 Question Type Distribution",
        "",
        "| question_type | count |",
        "|---------------|------:|",
    ]
    for qtype, count in v1_qa_stats["question_type_distribution"].items():
        lines.append(f"| {qtype} | {count} |")

    lines.extend(["", "### V2 Question Type Distribution", "", "| question_type | count |", "|---------------|------:|"])
    for qtype, count in v2_qa_stats["question_type_distribution"].items():
        lines.append(f"| {qtype} | {count} |")

    lines.extend(["", "## Retrieval Metrics (Dense, top_k=10)", "", "| Metric | V1 | V2 | Abs Δ | Rel Δ |", "|--------|---:|---:|------:|------:|"])
    for name in metric_names:
        v1 = v1_metrics.get(name, 0.0)
        v2 = v2_metrics.get(name, 0.0)
        lines.append(
            f"| {name} | {v1:.4f} | {v2:.4f} | {_abs_delta(v2, v1)} | {_pct_delta(v2, v1)} |"
        )

    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            f"- V2 Recall@10: **{v2_metrics.get('Recall@10', 0):.2%}** vs V1 **{v1_metrics.get('Recall@10', 0):.2%}**",
            f"- V2 MRR: **{v2_metrics.get('MRR', 0):.4f}** vs V1 **{v1_metrics.get('MRR', 0):.4f}**",
            "",
            "The retriever implementation was not modified. Differences reflect benchmark",
            "question quality (template navigation vs natural user information needs).",
            "",
            "## V2 Failure Analysis Summary",
            "",
            f"- Total failures (Recall@10=0): **{v2_failures.get('failure_count', 0)}**",
            "",
            "### Failure Patterns",
            "",
            "| Pattern | Count |",
            "|---------|------:|",
        ]
    )
    for pattern, count in v2_failures.get("failure_patterns", {}).items():
        lines.append(f"| {pattern} | {count} |")

    lines.extend(["", "### Top 20 Most Difficult Questions (V2)", ""])
    for i, item in enumerate(v2_failures.get("top_difficult", []), 1):
        lines.append(f"{i}. **{item['pattern']}** — {item['question'][:100]}")
        lines.append(f"   - gold: `{item['gold_unit_ids'][0] if item['gold_unit_ids'] else ''}`")
        if item.get("top_retrieved"):
            lines.append(f"   - top retrieved: `{item['top_retrieved'][0]}`")
        lines.append("")

    lines.extend(["### Most Frequently Retrieved Incorrect Documents (V2 failures)", ""])
    for doc, count in v2_failures.get("top_wrong_documents", []):
        lines.append(f"- `{doc}`: {count} times")

    lines.append("")
    return "\n".join(lines)


def render_statistics_report(
    *,
    v2_qa_stats: dict[str, Any],
    v2_stats: dict[str, Any],
    v2_failures: dict[str, Any],
) -> str:
    lines = [
        "# Retrieval Statistics (V2 Natural-Language Benchmark)",
        "",
        "## Dataset",
        "",
        f"- QA pairs: **{v2_qa_stats['count']}**",
        f"- Unique documents: **{v2_qa_stats['unique_documents']}**",
        "",
        "## Question Type Distribution",
        "",
        "| question_type | count | Recall@10 |",
        "|---------------|------:|----------:|",
    ]
    qtype_dist = v2_qa_stats["question_type_distribution"]
    qtype_r10 = v2_stats.get("question_type_recall_at_10", {})
    for qtype in sorted(qtype_dist.keys()):
        lines.append(f"| {qtype} | {qtype_dist[qtype]} | {qtype_r10.get(qtype, 0):.4f} |")

    lines.extend(
        [
            "",
            "## Document-Level Recall@10",
            "",
            "| document_id | Recall@10 |",
            "|-------------|----------:|",
        ]
    )
    for doc, r10 in sorted(v2_stats.get("document_recall_at_10", {}).items()):
        lines.append(f"| {doc} | {r10:.4f} |")

    lines.extend(
        [
            "",
            "## Score Statistics",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Average retrieved score (all Top-10) | {v2_stats.get('avg_retrieved_score', 0):.4f} |",
            f"| Average Top-1 score (successful Recall@10) | {v2_stats.get('avg_success_top1_score', 0):.4f} |",
            f"| Average Top-1 score (failed Recall@10) | {v2_stats.get('avg_failure_top1_score', 0):.4f} |",
            "",
            "## Top-10 Most Frequently Retrieved Documents",
            "",
            "| document_id | times in Top-10 |",
            "|-------------|----------------:|",
        ]
    )
    for doc, count in v2_stats.get("top_retrieved_documents", []):
        lines.append(f"| {doc} | {count} |")

    lines.extend(["", "## Failure Patterns", ""])
    for pattern, count in v2_failures.get("failure_patterns", {}).items():
        lines.append(f"- **{pattern}**: {count}")

    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare V1/V2 QA benchmarks and retrieval results.")
    parser.add_argument("--v1-qa", default="data/qa_dataset/qa_pairs_v1.jsonl")
    parser.add_argument("--v2-qa", default="data/qa_dataset/qa_pairs_v2.jsonl")
    parser.add_argument("--v1-results", default="results/retrieval/dense/retrieval_results_v1.jsonl")
    parser.add_argument("--v1-matched-results", default="results/retrieval/dense/retrieval_results_v1_matched.jsonl")
    parser.add_argument("--v2-results", default="results/retrieval/dense/retrieval_results_v2.jsonl")
    parser.add_argument(
        "--comparison-report",
        default="results/ablation/qa_v2_dense_comparison.md",
    )
    parser.add_argument(
        "--statistics-report",
        default="results/retrieval/dense/retrieval_statistics.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    v1_qa = load_qa_jsonl(Path(args.v1_qa))
    v2_qa = load_qa_jsonl(Path(args.v2_qa))
    v1_results = load_retrieval_jsonl(Path(args.v1_results))
    v1_matched_results = load_retrieval_jsonl(Path(args.v1_matched_results))
    v2_results = load_retrieval_jsonl(Path(args.v2_results))

    v1_qa_stats = qa_dataset_stats(v1_qa)
    v2_qa_stats = qa_dataset_stats(v2_qa)
    v1_metrics = retrieval_summary(v1_results)
    v1_matched_metrics = retrieval_summary(v1_matched_results)
    v2_metrics = retrieval_summary(v2_results)
    v2_failures = analyze_failures(v2_results)
    v2_stats = retrieval_statistics(v2_results)

    comparison = render_comparison_report(
        v1_qa_stats=v1_qa_stats,
        v2_qa_stats=v2_qa_stats,
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        v2_failures=v2_failures,
        v1_path=args.v1_qa,
        v2_path=args.v2_qa,
    )
    if v1_matched_metrics:
        matched_lines = [
            "",
            "## Matched-Unit Comparison (same gold unit_ids)",
            "",
            f"V1 matched subset: **{len(v1_matched_results)}** pairs | V2: **{len(v2_results)}** pairs",
            "",
            "| Metric | V1 (matched) | V2 | Abs Δ |",
            "|--------|-------------:|---:|------:|",
        ]
        for name in ["Recall@1", "Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]:
            v1m = v1_matched_metrics.get(name, 0.0)
            v2m = v2_metrics.get(name, 0.0)
            matched_lines.append(f"| {name} | {v1m:.4f} | {v2m:.4f} | {_abs_delta(v2m, v1m)} |")
        comparison += "\n".join(matched_lines)
    statistics = render_statistics_report(
        v2_qa_stats=v2_qa_stats,
        v2_stats=v2_stats,
        v2_failures=v2_failures,
    )

    comp_path = Path(args.comparison_report)
    stat_path = Path(args.statistics_report)
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(comparison + "\n", encoding="utf-8")
    stat_path.write_text(statistics + "\n", encoding="utf-8")

    print(f"Wrote {comp_path}")
    print(f"Wrote {stat_path}")
    if v2_metrics:
        print(
            f"V2: Recall@10={v2_metrics.get('Recall@10', 0):.4f} "
            f"MRR={v2_metrics.get('MRR', 0):.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
