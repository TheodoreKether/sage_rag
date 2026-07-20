"""Run paper-quality root-cause analysis on latest retrieval failures."""

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

from src.analysis.failure_classifier import UnitInfo
from src.analysis.root_cause_classifier import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    RootCauseRecord,
    SAGE_SOLUTIONS,
    classify_failure,
)

QA_PATH = ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
UNITS_PATH = ROOT / "data/evidence_units/evidence_units.jsonl"
OUTPUT_DIR = ROOT / "results/root_cause_analysis"

RETRIEVER_PATHS = {
    "bm25": ROOT / "results/retrieval/bm25/retrieval_results.jsonl",
    "dense": ROOT / "results/retrieval/dense/retrieval_results_v2.jsonl",
    "hybrid": ROOT / "results/retrieval/hybrid/retrieval_results.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_units(path: Path) -> dict[str, UnitInfo]:
    return {u.unit_id: u for u in (UnitInfo.from_record(r) for r in load_jsonl(path))}


def format_snippet(unit_id: str, unit_index: dict[str, UnitInfo], limit: int = 140) -> str:
    u = unit_index.get(unit_id)
    if not u:
        return unit_id
    text = u.text.replace("\n", " ").strip()
    if len(text) > limit:
        text = text[:limit] + "..."
    return f"{unit_id} — {text}"


def build_stats(
    failures: list[RootCauseRecord],
    *,
    total_queries: int,
    total_failures: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter(f.primary_category for f in failures)
    examples: dict[str, list[str]] = defaultdict(list)
    subtypes: dict[str, Counter[str]] = defaultdict(Counter)
    for f in failures:
        if len(examples[f.primary_category]) < 8:
            examples[f.primary_category].append(f.qa_id)
        if f.hierarchical_subtype:
            subtypes[f.primary_category][f.hierarchical_subtype] += 1

    n_fail = total_failures or 1
    n_true = sum(1 for f in failures if not f.is_dataset_issue) or 1
    categories: dict[str, Any] = {}
    for cat in CATEGORY_ORDER:
        c = counts.get(cat, 0)
        entry: dict[str, Any] = {
            "count": c,
            "percentage_of_failures": round(100.0 * c / n_fail, 2),
            "percentage_of_all_queries": round(100.0 * c / total_queries, 2),
            "examples": examples.get(cat, []),
        }
        if cat != "dataset_annotation_issue":
            entry["percentage_of_retriever_failures"] = round(
                100.0 * c / n_true if not failures else 100.0 * c / max(n_true, 1),
                2,
            )
        if subtypes.get(cat):
            entry["subtypes"] = dict(subtypes[cat])
        categories[cat] = entry

    dataset_n = counts.get("dataset_annotation_issue", 0)
    return {
        "total_failures": total_failures,
        "dataset_issues": dataset_n,
        "retriever_failures": total_failures - dataset_n,
        "failure_rate_pct": round(100.0 * total_failures / total_queries, 2),
        "categories": categories,
    }


def pick_examples(
    failures: list[RootCauseRecord],
    per_category: int = 5,
) -> dict[str, list[RootCauseRecord]]:
    by_cat: dict[str, list[RootCauseRecord]] = defaultdict(list)
    for f in failures:
        by_cat[f.primary_category].append(f)
    selected: dict[str, list[RootCauseRecord]] = {}
    for cat, items in by_cat.items():
        picked: list[RootCauseRecord] = []
        seen: set[str] = set()
        for item in items:
            key = item.root_cause[:100]
            if key in seen:
                continue
            seen.add(key)
            picked.append(item)
            if len(picked) >= per_category:
                break
        for item in items:
            if len(picked) >= per_category:
                break
            if item not in picked:
                picked.append(item)
        selected[cat] = picked
    return selected


def write_statistics_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_root_cause_analysis_md(
    payload: dict[str, Any],
    hybrid_examples: dict[str, list[RootCauseRecord]],
    unit_index: dict[str, UnitInfo],
    path: Path,
) -> None:
    hybrid = payload["retrievers"]["hybrid"]
    lines = [
        "# Root Cause Analysis of Retrieval Failures",
        "",
        "Paper-oriented analysis of **Recall@10 misses** on QA Dataset V2 "
        "(aligned/rewritten), using the latest BM25 / Dense / Hybrid evaluation outputs.",
        "",
        "**Failure definition:** gold evidence unit is absent from Top-10.",
        "",
        "## Setup",
        "",
        f"- QA: `data/qa_dataset/qa_pairs_v2.jsonl` ({payload['total_queries']} queries)",
        "- Results: `results/retrieval/{bm25,dense,hybrid}/`",
        "- Primary focus: **Hybrid** (final fusion baseline)",
        "",
        "## Summary",
        "",
        "| Retriever | Failures | Failure Rate | Dataset Issues | True Retriever Failures |",
        "|-----------|----------:|-------------:|---------------:|------------------------:|",
    ]
    for name in ("bm25", "dense", "hybrid"):
        s = payload["retrievers"][name]
        lines.append(
            f"| {name.upper()} | {s['total_failures']} | {s['failure_rate_pct']:.2f}% | "
            f"{s['dataset_issues']} | {s['retriever_failures']} |"
        )

    lines += [
        "",
        "## Dataset Issue",
        "",
        "Before attributing misses to retrieval, we filter question–gold annotation problems.",
        "",
        f"- **Hybrid count:** {hybrid['categories']['dataset_annotation_issue']['count']}",
        f"- **Share of Hybrid misses:** "
        f"{hybrid['categories']['dataset_annotation_issue']['percentage_of_failures']:.1f}%",
        "",
    ]
    for ex in hybrid_examples.get("dataset_annotation_issue", [])[:3]:
        lines += [
            f"### Case `{ex.qa_id}`",
            "",
            f"- **Question:** {ex.question}",
            f"- **Gold:** {format_snippet(ex.gold_unit_ids[0], unit_index)}",
            f"- **Root cause:** {ex.root_cause}",
            "",
        ]

    for cat in CATEGORY_ORDER:
        if cat == "dataset_annotation_issue":
            continue
        st = hybrid["categories"].get(cat, {})
        label = CATEGORY_LABELS[cat]
        lines += [
            f"## {label}",
            "",
            f"- **Count:** {st.get('count', 0)}",
            f"- **% of Hybrid misses:** {st.get('percentage_of_failures', 0):.1f}%",
            f"- **% of true retriever failures:** {st.get('percentage_of_retriever_failures', 0):.1f}%",
            "",
        ]
        if st.get("subtypes"):
            lines.append(f"- **Subtypes:** `{st['subtypes']}`")
            lines.append("")
        for ex in hybrid_examples.get(cat, [])[:2]:
            lines += [
                f"### Case `{ex.qa_id}`",
                "",
                f"- **Question:** {ex.question}",
                f"- **Gold:** {format_snippet(ex.gold_unit_ids[0], unit_index)}",
                f"- **Top-3:** {', '.join(ex.retrieved_unit_ids[:3])}",
                f"- **Root cause:** {ex.root_cause}",
                "",
            ]

    lines += [
        "## Key Takeaways",
        "",
        "1. Dataset issues must be separated from retriever limitations for fair method comparison.",
        "2. Remaining Hybrid misses are dominated by structural and semantic phenomena "
        "(appendix, cross-reference, version, table, hierarchy, residual semantic confusion).",
        "3. These residual failures motivate structure-aware SAGE-RAG modules rather than "
        "further rank fusion alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_examples_md(
    hybrid_examples: dict[str, list[RootCauseRecord]],
    unit_index: dict[str, UnitInfo],
    path: Path,
) -> None:
    lines = [
        "# Failure Examples (Hybrid Recall@10 Misses)",
        "",
        "Representative cases per root-cause category (≥5 when available).",
        "",
    ]
    idx = 1
    for cat in CATEGORY_ORDER:
        items = hybrid_examples.get(cat, [])
        if not items:
            continue
        lines += [f"# {CATEGORY_LABELS[cat]}", ""]
        for ex in items:
            lines += [
                f"## Example {idx}",
                "",
                f"**Question:** {ex.question}",
                "",
                "**Gold Evidence:**",
            ]
            for gid in ex.gold_unit_ids[:2]:
                lines.append(f"- {format_snippet(gid, unit_index, limit=220)}")
            lines += ["", "**Retrieved Top10:**"]
            for rid in ex.retrieved_unit_ids[:10]:
                lines.append(f"- {format_snippet(rid, unit_index, limit=100)}")
            lines += [
                "",
                f"**Failure Type:** {CATEGORY_LABELS[cat]}",
                "",
                f"**Root Cause:** {ex.root_cause}",
                "",
                f"**Why Retriever Failed:** {ex.why_failed}",
                "",
                f"**Potential SAGE-RAG Solution:** {ex.potential_solution}",
                "",
            ]
            idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def write_design_motivation(payload: dict[str, Any], path: Path) -> None:
    hybrid = payload["retrievers"]["hybrid"]["categories"]
    bm25 = payload["retrievers"]["bm25"]["categories"]
    dense = payload["retrievers"]["dense"]["categories"]

    def c(stats: dict, key: str) -> int:
        return int(stats.get(key, {}).get("count", 0))

    lines = [
        "# Design Motivation from Root Cause Analysis",
        "",
        "Evidence-only motivation for SAGE-RAG structural modules. "
        "**No algorithms proposed.** Based on Hybrid-focused root-cause analysis "
        "of Recall@10 misses on aligned QA V2.",
        "",
        "## What Flat Retrieval Cannot Solve",
        "",
        f"- Even after Dense+BM25 RRF fusion, Hybrid still misses "
        f"**{payload['retrievers']['hybrid']['retriever_failures']}** queries "
        f"(excluding {payload['retrievers']['hybrid']['dataset_issues']} dataset issues).",
        "- Rank fusion improves over Dense but does not remove structural failure modes.",
        "",
        "## Which Failures Need Structural Information",
        "",
        "| Observed Failure | Count (Hybrid) | Structural Signal Needed |",
        "|------------------|---------------:|--------------------------|",
        f"| Hierarchical Structure | {c(hybrid, 'hierarchical_structure_failure')} | Clause tree (parent/child/sibling) |",
        f"| Cross-reference | {c(hybrid, 'cross_reference_failure')} | Explicit reference edges |",
        f"| Table Information | {c(hybrid, 'table_information_failure')} | Table nodes + captions |",
        f"| Appendix | {c(hybrid, 'appendix_failure')} | Body↔annex links |",
        f"| Version Confusion | {c(hybrid, 'version_confusion')} | Document family / edition metadata |",
        f"| Multi-clause Reasoning | {c(hybrid, 'multi_clause_reasoning_failure')} | Multi-hop clause graph |",
        f"| Residual Semantic Similarity | {c(hybrid, 'semantic_similarity_failure')} | Structure-aware disambiguation |",
        f"| Lexical Matching (BM25-side) | {c(bm25, 'lexical_matching_failure')} | Semantic complement to sparse matching |",
        "",
        "## Module Motivations",
        "",
        "### Hierarchical Graph",
        "",
        f"- Hybrid hierarchical failures: **{c(hybrid, 'hierarchical_structure_failure')}**.",
        "- Flat Top-K often returns a parent, child, or sibling clause instead of the gold node.",
        "- This shows that clause hierarchy is an unused signal in current baselines.",
        "",
        "### Cross-reference Graph",
        "",
        f"- Hybrid cross-reference failures: **{c(hybrid, 'cross_reference_failure')}**.",
        "- Gold units contain pointers such as “见条款 / See Annex / 如下表”, which retrieval does not traverse.",
        "- Modeling reference relations is required for multi-hop evidence discovery.",
        "",
        "### Table Nodes",
        "",
        f"- Hybrid table failures: **{c(hybrid, 'table_information_failure')}**.",
        "- Answers grounded in tables are under-recalled by flat text units.",
        "- Explicit table–evidence relations are needed.",
        "",
        "### Appendix Links",
        "",
        f"- Hybrid appendix failures: **{c(hybrid, 'appendix_failure')}**.",
        "- Annex-located gold is systematically hard for both lexical and dense flat retrieval.",
        "- Normative body → appendix linkage is a structural necessity.",
        "",
        "### Document / Version Metadata",
        "",
        f"- Hybrid version confusion: **{c(hybrid, 'version_confusion')}**.",
        f"- Dense shows more version confusion ({c(dense, 'version_confusion')}) than BM25 "
        f"({c(bm25, 'version_confusion')}), indicating embedding similarity across editions.",
        "",
        "## Sparse vs Dense vs Graph Need",
        "",
        f"- **Sparse (BM25) limitation:** lexical gaps "
        f"({c(bm25, 'lexical_matching_failure')} BM25 misses where Dense succeeds).",
        f"- **Dense limitation:** semantic near-misses and version/cross-doc confusion "
        f"(semantic {c(dense, 'semantic_similarity_failure')}, version {c(dense, 'version_confusion')}).",
        "- **Graph need:** hierarchy, cross-reference, appendix, and table failures persist after Hybrid "
        "fusion — these are not fixed by score combination alone.",
        "",
        "## Dataset Hygiene Note",
        "",
        f"- Hybrid dataset/annotation issues filtered: "
        f"**{payload['retrievers']['hybrid']['dataset_issues']}**.",
        "- These must not be counted as method failures when comparing SAGE-RAG to baselines.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_comparison(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Failure Type Comparison: BM25 vs Dense vs Hybrid",
        "",
        "Counts are Recall@10 misses on aligned QA V2, after the same root-cause taxonomy.",
        "",
        "## Counts by Failure Type",
        "",
        "| Failure Type | BM25 | Dense | Hybrid |",
        "|--------------|-----:|------:|-------:|",
    ]
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS[cat]
        b = payload["retrievers"]["bm25"]["categories"][cat]["count"]
        d = payload["retrievers"]["dense"]["categories"][cat]["count"]
        h = payload["retrievers"]["hybrid"]["categories"][cat]["count"]
        lines.append(f"| {label} | {b} | {d} | {h} |")

    lines += [
        "",
        "## % of Each Retriever's Misses",
        "",
        "| Failure Type | BM25 % | Dense % | Hybrid % |",
        "|--------------|-------:|--------:|---------:|",
    ]
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS[cat]
        row = []
        for name in ("bm25", "dense", "hybrid"):
            row.append(
                f"{payload['retrievers'][name]['categories'][cat]['percentage_of_failures']:.1f}"
            )
        lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "### Sparse Retrieval (BM25) limitations",
        "",
        "- Higher **Lexical Matching Failure** when Dense succeeds: synonym / paraphrase gaps.",
        "- Stronger on exact technical terms; still fails on appendix/table/version structure.",
        "",
        "### Dense Retrieval limitations",
        "",
        "- Higher **Semantic Similarity Failure** and often higher **Version Confusion**.",
        "- Embeddings pull topically related but structurally wrong clauses/documents.",
        "",
        "### What remains after Hybrid (needs Graph-based Retrieval)",
        "",
        "- Hybrid reduces some Dense errors but **Appendix / Cross-reference / Table / Hierarchical / Version** "
        "failures remain.",
        "- These residual classes are the experimental justification for SAGE-RAG structural graphs "
        "rather than additional flat fusion.",
        "",
        "### Dataset issues",
        "",
        "- Present across all retrievers; should be excluded when claiming method gains.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_retriever(
    name: str,
    records: dict[str, dict[str, Any]],
    unit_index: dict[str, UnitInfo],
    qa_index: dict[str, dict[str, Any]],
    hit_maps: dict[str, dict[str, bool]],
) -> list[RootCauseRecord]:
    out: list[RootCauseRecord] = []
    for qa_id, rec in records.items():
        fr = classify_failure(
            rec,
            unit_index=unit_index,
            qa_index=qa_index,
            analyzing_retriever=name,
            dense_hit=hit_maps["dense"].get(qa_id),
            bm25_hit=hit_maps["bm25"].get(qa_id),
        )
        if fr:
            out.append(fr)
    return out


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_index = load_units(UNITS_PATH)
    qa_index = {r["qa_id"]: r for r in load_jsonl(QA_PATH)}
    total_queries = len(qa_index)

    retriever_records: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in RETRIEVER_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing retrieval results: {path}")
        retriever_records[name] = {r["qa_id"]: r for r in load_jsonl(path)}

    hit_maps = {
        name: {
            qa_id: float(rec.get("Recall@10", 0)) > 0
            for qa_id, rec in records.items()
        }
        for name, records in retriever_records.items()
    }

    failures_by: dict[str, list[RootCauseRecord]] = {}
    for name in ("bm25", "dense", "hybrid"):
        failures_by[name] = analyze_retriever(
            name,
            retriever_records[name],
            unit_index,
            qa_index,
            hit_maps,
        )

    payload: dict[str, Any] = {
        "dataset": "qa_pairs_v2 (aligned)",
        "qa_path": str(QA_PATH.relative_to(ROOT)),
        "total_queries": total_queries,
        "failure_definition": "Recall@10 == 0",
        "retrievers": {},
    }
    for name, fails in failures_by.items():
        payload["retrievers"][name] = build_stats(
            fails,
            total_queries=total_queries,
            total_failures=len(fails),
        )

    # Flatten hybrid categories for the schema requested in the prompt
    hybrid_flat = {
        cat: {
            "failure_type": CATEGORY_LABELS[cat],
            "count": info["count"],
            "percentage": info["percentage_of_failures"],
            "examples": info["examples"],
        }
        for cat, info in payload["retrievers"]["hybrid"]["categories"].items()
    }
    payload["hybrid_root_causes"] = hybrid_flat

    write_statistics_json(payload, output_dir / "root_cause_statistics.json")

    hybrid_examples = pick_examples(failures_by["hybrid"], per_category=5)
    write_root_cause_analysis_md(
        payload,
        hybrid_examples,
        unit_index,
        output_dir / "root_cause_analysis.md",
    )
    write_failure_examples_md(
        hybrid_examples,
        unit_index,
        output_dir / "failure_examples.md",
    )
    write_design_motivation(payload, output_dir / "design_motivation.md")
    write_failure_comparison(payload, output_dir / "failure_comparison.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Root-cause analysis of retrieval failures")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    payload = run(args.output_dir)
    print(f"Root-cause analysis written to {args.output_dir}")
    for name, summary in payload["retrievers"].items():
        print(
            f"  {name}: failures={summary['total_failures']} "
            f"(dataset={summary['dataset_issues']}, "
            f"retriever={summary['retriever_failures']})"
        )


if __name__ == "__main__":
    main()
