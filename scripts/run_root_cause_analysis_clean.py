"""Root-cause analysis on Clean Benchmark failures (no Dataset Issue category)."""

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
    RootCauseRecord,
    classify_retrieval_root_cause,
    detect_dataset_issue,
)

QA_CLEAN = ROOT / "data/qa_dataset/qa_pairs_clean.jsonl"
UNITS = ROOT / "data/evidence_units/evidence_units.jsonl"
OUT_DIR = ROOT / "results/root_cause_analysis_clean"

RETRIEVER_PATHS = {
    "bm25": ROOT / "results/retrieval/clean_benchmark/bm25_results.jsonl",
    "dense": ROOT / "results/retrieval/clean_benchmark/dense_results.jsonl",
    "hybrid": ROOT / "results/retrieval/clean_benchmark/hybrid_results.jsonl",
}

# Clean taxonomy (paper-facing labels)
CLEAN_ORDER = [
    "document_identity_version_disambiguation",
    "appendix_retrieval_failure",
    "cross_reference_failure",
    "hierarchy_structure_failure",
    "table_structured_content_failure",
    "semantic_misunderstanding",
    "lexical_mismatch",
]

CLEAN_LABELS = {
    "document_identity_version_disambiguation": "Document Identity / Version Disambiguation",
    "appendix_retrieval_failure": "Appendix Retrieval Failure",
    "cross_reference_failure": "Cross-reference Failure",
    "hierarchy_structure_failure": "Hierarchy Structure Failure",
    "table_structured_content_failure": "Table / Structured Content Failure",
    "semantic_misunderstanding": "Semantic Misunderstanding",
    "lexical_mismatch": "Lexical Mismatch",
}

# Map internal classifier categories → clean taxonomy
INTERNAL_TO_CLEAN = {
    "version_confusion": "document_identity_version_disambiguation",
    "appendix_failure": "appendix_retrieval_failure",
    "cross_reference_failure": "cross_reference_failure",
    "hierarchical_structure_failure": "hierarchy_structure_failure",
    "table_information_failure": "table_structured_content_failure",
    "semantic_similarity_failure": "semantic_misunderstanding",
    "lexical_matching_failure": "lexical_mismatch",
    "multi_clause_reasoning_failure": "cross_reference_failure",
}

SAGE_MAP = {
    "document_identity_version_disambiguation": "Document Version Relation",
    "appendix_retrieval_failure": "Appendix–Evidence Relation",
    "cross_reference_failure": "Reference Relation",
    "hierarchy_structure_failure": "Clause Hierarchy Relation",
    "table_structured_content_failure": "Table–Evidence Relation",
    "semantic_misunderstanding": "Structure-aware disambiguation beyond flat similarity",
    "lexical_mismatch": "Complement sparse matching with semantic / structural signals",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_snippet(unit_id: str, unit_index: dict[str, UnitInfo], limit: int = 160) -> str:
    u = unit_index.get(unit_id)
    if not u:
        return unit_id
    text = u.text.replace("\n", " ").strip()
    if len(text) > limit:
        text = text[:limit] + "..."
    return f"{unit_id} — {text}"


def classify_clean_failure(
    rec: dict[str, Any],
    *,
    unit_index: dict[str, UnitInfo],
    qa_index: dict[str, dict[str, Any]],
    analyzing_retriever: str,
    dense_hit: bool | None,
    bm25_hit: bool | None,
) -> RootCauseRecord | None:
    if float(rec.get("Recall@10", 0.0)) > 0:
        return None
    gold_ids = list(rec.get("gold_unit_ids") or [])
    retrieved_ids = list(rec.get("retrieved_unit_ids") or [])
    if not gold_ids:
        return None

    qa = qa_index.get(str(rec.get("qa_id", "")), {})
    gold = unit_index.get(gold_ids[0])
    if gold is None:
        return None

    question = str(rec.get("question") or qa.get("question") or "")
    qtype = str(rec.get("question_type") or qa.get("question_type") or "")

    # Safety: skip residual dataset issues (should already be removed from clean QA)
    is_issue, reason = detect_dataset_issue(question, qtype, gold, doc_title=gold.title or "")
    if is_issue:
        return None

    cat, subtype, root, why, secondary = classify_retrieval_root_cause(
        gold=gold,
        retrieved_ids=retrieved_ids,
        unit_index=unit_index,
        gold_unit_ids=gold_ids,
        dense_hit=dense_hit,
        bm25_hit=bm25_hit,
        analyzing_retriever=analyzing_retriever,
    )
    clean_cat = INTERNAL_TO_CLEAN.get(cat, "semantic_misunderstanding")
    return RootCauseRecord(
        qa_id=str(rec.get("qa_id", "")),
        question=question,
        question_type=qtype,
        document_id=str(rec.get("document_id") or gold.document_id),
        gold_unit_ids=gold_ids,
        retrieved_unit_ids=retrieved_ids,
        primary_category=clean_cat,
        root_cause=root,
        why_failed=why,
        potential_solution=SAGE_MAP.get(clean_cat, ""),
        gold_text=gold.text[:500],
        hierarchical_subtype=subtype,
        is_dataset_issue=False,
        secondary_signals=secondary,
        dense_hit=dense_hit,
        bm25_hit=bm25_hit,
    )


def build_stats(failures: list[RootCauseRecord], total_queries: int) -> dict[str, Any]:
    counts = Counter(f.primary_category for f in failures)
    examples: dict[str, list[str]] = defaultdict(list)
    for f in failures:
        if len(examples[f.primary_category]) < 10:
            examples[f.primary_category].append(f.qa_id)
    n = len(failures) or 1
    cats = {}
    for cat in CLEAN_ORDER:
        c = counts.get(cat, 0)
        cats[cat] = {
            "failure_type": CLEAN_LABELS[cat],
            "count": c,
            "percentage": round(100.0 * c / n, 2),
            "percentage_of_all_queries": round(100.0 * c / total_queries, 2),
            "examples": examples.get(cat, []),
        }
    return {
        "total_failures": len(failures),
        "failure_rate_pct": round(100.0 * len(failures) / total_queries, 2),
        "categories": cats,
    }


def pick_examples(
    failures: list[RootCauseRecord], per_category: int = 5
) -> dict[str, list[RootCauseRecord]]:
    by_cat: dict[str, list[RootCauseRecord]] = defaultdict(list)
    for f in failures:
        by_cat[f.primary_category].append(f)
    selected: dict[str, list[RootCauseRecord]] = {}
    for cat in CLEAN_ORDER:
        items = by_cat.get(cat, [])
        picked: list[RootCauseRecord] = []
        seen: set[str] = set()
        for item in items:
            key = item.root_cause[:90]
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


def write_statistics_md(payload: dict[str, Any], path: Path) -> None:
    hybrid = payload["retrievers"]["hybrid"]
    lines = [
        "# Root Cause Analysis (Clean Benchmark)",
        "",
        "Failures on `qa_pairs_clean.jsonl` after removing Dataset / Annotation Issues.",
        "",
        f"- Total clean queries: **{payload['total_queries']}**",
        f"- Hybrid failures (Recall@10 miss): **{hybrid['total_failures']}** "
        f"({hybrid['failure_rate_pct']:.2f}%)",
        "",
        "## Hybrid Failure Distribution",
        "",
        "| Failure Type | Count | Percentage |",
        "|--------------|------:|-----------:|",
    ]
    for cat in CLEAN_ORDER:
        st = hybrid["categories"][cat]
        lines.append(
            f"| {st['failure_type']} | {st['count']} | {st['percentage']:.1f}% |"
        )

    lines += ["", "## Comparison Across Retrievers", "", "| Failure Type | BM25 | Dense | Hybrid |", "|--------------|-----:|------:|-------:|"]
    for cat in CLEAN_ORDER:
        label = CLEAN_LABELS[cat]
        b = payload["retrievers"]["bm25"]["categories"][cat]["count"]
        d = payload["retrievers"]["dense"]["categories"][cat]["count"]
        h = payload["retrievers"]["hybrid"]["categories"][cat]["count"]
        lines.append(f"| {label} | {b} | {d} | {h} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_examples_md(
    examples: dict[str, list[RootCauseRecord]],
    unit_index: dict[str, UnitInfo],
    path: Path,
) -> None:
    lines = [
        "# Clean Benchmark Failure Examples",
        "",
        "Representative Hybrid Recall@10 misses (Dataset Issues excluded).",
        "",
    ]
    idx = 1
    for cat in CLEAN_ORDER:
        items = examples.get(cat, [])
        if not items:
            continue
        lines += [f"# {CLEAN_LABELS[cat]}", ""]
        for ex in items:
            lines += [
                f"## Example {idx}",
                "",
                f"**Question:** {ex.question}",
                "",
                "**Gold Evidence:**",
                f"- {format_snippet(ex.gold_unit_ids[0], unit_index, 220)}",
                "",
                "**Retrieved Results (Top-10):**",
            ]
            for rid in ex.retrieved_unit_ids[:10]:
                lines.append(f"- {format_snippet(rid, unit_index, 100)}")
            lines += [
                "",
                f"**Failure Type:** {CLEAN_LABELS[cat]}",
                "",
                f"**Root Cause:** {ex.root_cause}",
                "",
                f"**Why Existing Retrieval Fails:** {ex.why_failed}",
                "",
                f"**Structural Signal Needed:** {ex.potential_solution}",
                "",
            ]
            idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def write_design_motivation(payload: dict[str, Any], path: Path) -> None:
    h = payload["retrievers"]["hybrid"]["categories"]

    def c(key: str) -> int:
        return int(h[key]["count"])

    lines = [
        "# SAGE-RAG Design Motivation (Clean Benchmark)",
        "",
        "Evidence-only draft for the paper section *Motivation and Problem Analysis*. "
        "**No algorithms proposed.** Based on Hybrid failures after Dataset Issue removal.",
        "",
        "## What BM25 / Dense / Hybrid Still Cannot Solve",
        "",
        f"On the clean benchmark, Hybrid still misses **{payload['retrievers']['hybrid']['total_failures']}** "
        f"queries ({payload['retrievers']['hybrid']['failure_rate_pct']:.2f}%). "
        "These residual errors persist after lexical–semantic fusion, indicating limits of flat retrieval.",
        "",
        "## Problem → Required Structural Information",
        "",
        "### Document Identity / Version Disambiguation",
        "",
        f"- Hybrid count: **{c('document_identity_version_disambiguation')}**",
        "- Flat retrieval confuses editions of the same standard family (e.g., ISO …-2021 vs …-2024).",
        "- **Needed:** Document Version Relation (family + edition metadata / links).",
        "",
        "### Appendix Retrieval Failure",
        "",
        f"- Hybrid count: **{c('appendix_retrieval_failure')}**",
        "- Gold evidence in Annex/Appendix is systematically under-recalled.",
        "- **Needed:** Appendix–Evidence Relation linking normative body to annex content.",
        "",
        "### Cross-reference Failure",
        "",
        f"- Hybrid count: **{c('cross_reference_failure')}**",
        "- Gold text contains pointers such as “见条款 / 参见附录 / See Clause”.",
        "- **Needed:** Reference Relation over citing and cited clauses.",
        "",
        "### Hierarchy Structure Failure",
        "",
        f"- Hybrid count: **{c('hierarchy_structure_failure')}**",
        "- Retrievers return parent/child/sibling clauses instead of the gold node.",
        "- **Needed:** Clause Hierarchy Relation (chapter → clause → sub-clause).",
        "",
        "### Table / Structured Content Failure",
        "",
        f"- Hybrid count: **{c('table_structured_content_failure')}**",
        "- Answers grounded in tables/lists are poorly handled by flat text chunks.",
        "- **Needed:** Table–Evidence Relation (table nodes + captions).",
        "",
        "### Semantic Misunderstanding",
        "",
        f"- Hybrid count: **{c('semantic_misunderstanding')}**",
        "- Semantically related but incorrect clauses remain after Hybrid fusion.",
        "- **Needed:** Structure-aware disambiguation beyond embedding/lexical similarity alone.",
        "",
        "### Lexical Mismatch",
        "",
        f"- Observed mainly on BM25 "
        f"({payload['retrievers']['bm25']['categories']['lexical_mismatch']['count']} misses "
        "where Dense succeeds).",
        "- Synonym / paraphrase gaps remain a sparse-retriever limitation.",
        "- Hybrid partially mitigates but does not replace the need for structural cues on other failure types.",
        "",
        "## Summary for Method Design",
        "",
        "Rank fusion (Hybrid) improves over Dense but **does not eliminate** version, appendix, "
        "cross-reference, hierarchy, or table failures. These classes motivate SAGE-RAG’s use of "
        "explicit structural relations rather than additional flat score combination.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_index = {}
    for r in load_jsonl(UNITS):
        u = UnitInfo.from_record(r)
        unit_index[u.unit_id] = u

    qa_index = {r["qa_id"]: r for r in load_jsonl(QA_CLEAN)}
    total_queries = len(qa_index)

    records_by = {}
    for name, path in RETRIEVER_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing clean retrieval results: {path}")
        records_by[name] = {r["qa_id"]: r for r in load_jsonl(path)}

    hit_maps = {
        name: {qid: float(rec.get("Recall@10", 0)) > 0 for qid, rec in recs.items()}
        for name, recs in records_by.items()
    }

    failures_by: dict[str, list[RootCauseRecord]] = {}
    for name in ("bm25", "dense", "hybrid"):
        fails: list[RootCauseRecord] = []
        for qid, rec in records_by[name].items():
            fr = classify_clean_failure(
                rec,
                unit_index=unit_index,
                qa_index=qa_index,
                analyzing_retriever=name,
                dense_hit=hit_maps["dense"].get(qid),
                bm25_hit=hit_maps["bm25"].get(qid),
            )
            if fr:
                fails.append(fr)
        failures_by[name] = fails

    payload: dict[str, Any] = {
        "dataset": "qa_pairs_clean",
        "total_queries": total_queries,
        "failure_definition": "Recall@10 == 0 (Dataset Issues excluded from benchmark)",
        "retrievers": {
            name: build_stats(fails, total_queries) for name, fails in failures_by.items()
        },
    }

    (output_dir / "root_cause_clean_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_statistics_md(payload, output_dir / "root_cause_clean_statistics.md")

    hybrid_examples = pick_examples(failures_by["hybrid"], per_category=5)
    write_examples_md(
        hybrid_examples, unit_index, output_dir / "root_cause_clean_examples.md"
    )
    write_design_motivation(payload, output_dir / "design_motivation.md")

    # comparison-style markdown alias requested
    write_statistics_md(payload, output_dir / "failure_comparison.md")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-benchmark root-cause analysis")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    payload = run(args.output_dir)
    print(f"Clean RCA written to {args.output_dir}")
    for name, s in payload["retrievers"].items():
        print(f"  {name}: failures={s['total_failures']} ({s['failure_rate_pct']}%)")


if __name__ == "__main__":
    main()
