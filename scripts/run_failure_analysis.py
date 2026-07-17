"""Run retrieval failure analysis and generate benchmark reports."""

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

import matplotlib.pyplot as plt

from src.analysis.failure_classifier import (
    FailureRecord,
    UnitInfo,
    classify_failure_record,
    detect_lexical_failures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETRIEVER_PATHS = {
    "bm25": PROJECT_ROOT / "results/retrieval/bm25/retrieval_results.jsonl",
    "dense": PROJECT_ROOT / "results/retrieval/dense/retrieval_results_v2.jsonl",
    "hybrid": PROJECT_ROOT / "results/retrieval/hybrid/retrieval_results.jsonl",
}

QA_PATH = PROJECT_ROOT / "data/qa_dataset/qa_pairs_v2.jsonl"
UNITS_PATH = PROJECT_ROOT / "data/evidence_units/evidence_units.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results/failure_analysis"

CATEGORY_LABELS = {
    "hierarchical_failure": "Hierarchical Failure",
    "cross_document_failure": "Cross-document Failure",
    "version_failure": "Version Failure",
    "table_failure": "Table Failure",
    "appendix_failure": "Appendix Failure",
    "cross_reference_failure": "Cross-reference Failure",
    "semantic_failure": "Semantic Failure",
    "lexical_failure": "Lexical Failure",
}

CATEGORY_ORDER = [
    "hierarchical_failure",
    "cross_document_failure",
    "version_failure",
    "table_failure",
    "appendix_failure",
    "cross_reference_failure",
    "semantic_failure",
    "lexical_failure",
]

SOLUTION_HINTS = {
    "hierarchical_failure": "Hierarchical Graph linking parent/child/sibling clauses",
    "cross_document_failure": "Document-level metadata and domain-aware routing",
    "version_failure": "Version-aware document indexing and canonical family links",
    "table_failure": "Table Nodes with structured table content and captions",
    "appendix_failure": "Appendix Links connecting normative text to annex content",
    "cross_reference_failure": "Cross-reference Graph resolving 'See Clause X' pointers",
    "semantic_failure": "Structural context + graph-augmented retrieval (SAGE-RAG)",
    "lexical_failure": "Hybrid semantic + lexical signals; query expansion for rare terms",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_units(path: Path) -> dict[str, UnitInfo]:
    index: dict[str, UnitInfo] = {}
    for row in load_jsonl(path):
        info = UnitInfo.from_record(row)
        index[info.unit_id] = info
    return index


def load_qa_index(path: Path) -> dict[str, dict[str, Any]]:
    return {row["qa_id"]: row for row in load_jsonl(path)}


def load_retriever_records(path: Path) -> dict[str, dict[str, Any]]:
    return {row["qa_id"]: row for row in load_jsonl(path)}


def analyze_retriever(
    name: str,
    records: dict[str, dict[str, Any]],
    unit_index: dict[str, UnitInfo],
    qa_index: dict[str, dict[str, Any]],
) -> list[FailureRecord]:
    failures: list[FailureRecord] = []
    for rec in records.values():
        fr = classify_failure_record(rec, unit_index, qa_index)
        if fr:
            failures.append(fr)
    return failures


def build_category_stats(
    failures: list[FailureRecord],
    total_queries: int,
) -> dict[str, dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    subtypes: dict[str, Counter[str]] = defaultdict(Counter)

    for f in failures:
        counts[f.primary_category] += 1
        if len(examples[f.primary_category]) < 8:
            examples[f.primary_category].append(f.qa_id)
        if f.hierarchical_subtype:
            subtypes[f.primary_category][f.hierarchical_subtype] += 1

    stats: dict[str, dict[str, Any]] = {}
    n_fail = len(failures) or 1
    for cat in CATEGORY_ORDER:
        if cat == "lexical_failure":
            continue
        c = counts.get(cat, 0)
        entry: dict[str, Any] = {
            "count": c,
            "percentage_of_failures": round(100.0 * c / n_fail, 2),
            "percentage_of_all_queries": round(100.0 * c / total_queries, 2),
            "example_ids": examples.get(cat, []),
        }
        if subtypes.get(cat):
            entry["subtypes"] = dict(subtypes[cat])
        stats[cat] = entry
    return stats


def pick_representative_examples(
    failures: list[FailureRecord],
    per_category: int = 2,
) -> dict[str, list[FailureRecord]]:
    by_cat: dict[str, list[FailureRecord]] = defaultdict(list)
    for f in failures:
        by_cat[f.primary_category].append(f)

    selected: dict[str, list[FailureRecord]] = {}
    for cat, items in by_cat.items():
        seen_reason: set[str] = set()
        picked: list[FailureRecord] = []
        for item in items:
            key = item.reason[:80]
            if key in seen_reason:
                continue
            seen_reason.add(key)
            picked.append(item)
            if len(picked) >= per_category:
                break
        if len(picked) < per_category:
            for item in items:
                if item not in picked:
                    picked.append(item)
                if len(picked) >= per_category:
                    break
        selected[cat] = picked
    return selected


def format_unit_snippet(unit_id: str, unit_index: dict[str, UnitInfo]) -> str:
    u = unit_index.get(unit_id)
    if not u:
        return unit_id
    text = u.text.replace("\n", " ").strip()
    if len(text) > 120:
        text = text[:120] + "..."
    return f"{unit_id} — {text}"


def write_failure_statistics_json(
    output: dict[str, Any],
    path: Path,
) -> None:
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def write_failure_statistics_md(
    output: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Retrieval Failure Statistics",
        "",
        "Analysis of Recall@10 misses on QA Dataset V2 (492 queries).",
        "",
        "## Summary by Retriever",
        "",
        "| Retriever | Failures | Failure Rate |",
        "|-----------|----------|--------------|",
    ]
    for name, summary in output["retrievers"].items():
        lines.append(
            f"| {name.upper()} | {summary['failure_count']} | "
            f"{summary['failure_rate_pct']:.2f}% |"
        )

    lines.extend(["", "## Failure Categories", ""])
    for name, summary in output["retrievers"].items():
        lines.append(f"### {name.upper()}")
        lines.append("")
        lines.append("| Category | Count | % of Failures | Example IDs |")
        lines.append("|----------|-------|---------------|-------------|")
        for cat in CATEGORY_ORDER:
            if cat == "lexical_failure":
                continue
            st = summary["categories"].get(cat, {})
            if not st:
                continue
            ex = ", ".join(st.get("example_ids", [])[:4])
            label = CATEGORY_LABELS[cat]
            lines.append(
                f"| {label} | {st['count']} | {st['percentage_of_failures']:.1f}% | {ex} |"
            )
        lines.append("")

    lex = output.get("lexical_failure_comparison", {})
    if lex:
        lines.extend(
            [
                "## Lexical Failure (Dense hit, BM25 miss)",
                "",
                f"- Count: **{lex['count']}** ({lex['percentage_of_all_queries']:.2f}% of all queries)",
                f"- Example IDs: {', '.join(lex.get('example_ids', [])[:6])}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_examples_md(
    examples_by_retriever: dict[str, dict[str, list[FailureRecord]]],
    unit_index: dict[str, UnitInfo],
    lexical_examples: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Retrieval Failure Examples",
        "",
        "Representative Recall@10 failures per category (Hybrid retriever unless noted).",
        "",
    ]

    primary_retriever = "hybrid" if "hybrid" in examples_by_retriever else next(iter(examples_by_retriever))
    examples = examples_by_retriever[primary_retriever]

    for cat in CATEGORY_ORDER:
        if cat == "lexical_failure":
            continue
        items = examples.get(cat, [])
        if not items:
            continue
        lines.append(f"## {CATEGORY_LABELS[cat]}")
        lines.append("")
        for i, ex in enumerate(items, 1):
            lines.append(f"### Example {i}: `{ex.qa_id}`")
            lines.append("")
            lines.append(f"**Question:** {ex.question}")
            lines.append("")
            lines.append("**Gold Evidence:**")
            for gid in ex.gold_unit_ids[:2]:
                lines.append(f"- {format_unit_snippet(gid, unit_index)}")
            lines.append("")
            lines.append("**Retrieved Top-10 (first 5 shown):**")
            for rid in ex.retrieved_unit_ids[:5]:
                lines.append(f"- {format_unit_snippet(rid, unit_index)}")
            lines.append("")
            lines.append(f"**Reason:** {ex.reason}")
            if ex.hierarchical_subtype:
                lines.append(f"**Subtype:** {ex.hierarchical_subtype}")
            lines.append("")
            lines.append(f"**Possible Solution:** {SOLUTION_HINTS.get(cat, 'TBD')}")
            lines.append("")

    if lexical_examples:
        lines.append("## Lexical Failure (Dense ✓, BM25 ✗)")
        lines.append("")
        for i, ex in enumerate(lexical_examples[:3], 1):
            lines.append(f"### Example {i}: `{ex['qa_id']}`")
            lines.append("")
            lines.append(f"**Question:** {ex['question']}")
            lines.append("")
            lines.append("**Gold:** " + ", ".join(ex.get("gold_unit_ids", [])))
            lines.append("")
            lines.append("**Dense Top-3:**")
            for uid in ex.get("dense_top3", []):
                lines.append(f"- {format_unit_snippet(uid, unit_index)}")
            lines.append("")
            lines.append("**BM25 Top-3:**")
            for uid in ex.get("bm25_top3", []):
                lines.append(f"- {format_unit_snippet(uid, unit_index)}")
            lines.append("")
            lines.append(
                "**Reason:** Keyword mismatch — dense captures semantics but BM25 misses rare/paraphrased terms."
            )
            lines.append("")
            lines.append(f"**Possible Solution:** {SOLUTION_HINTS['lexical_failure']}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_failure_distribution(
    hybrid_stats: dict[str, dict[str, Any]],
    lexical_count: int,
    path: Path,
) -> None:
    labels: list[str] = []
    counts: list[int] = []
    for cat in CATEGORY_ORDER:
        if cat == "lexical_failure":
            c = lexical_count
        else:
            c = hybrid_stats.get(cat, {}).get("count", 0)
        if c > 0:
            labels.append(CATEGORY_LABELS[cat])
            counts.append(c)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.Set2(range(len(labels)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_ylabel("Failure Count")
    ax.set_title("Retrieval Failure Distribution (Hybrid Recall@10 Misses + Lexical)")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_design_motivation(
    output: dict[str, Any],
    path: Path,
) -> None:
    hybrid = output["retrievers"]["hybrid"]["categories"]
    bm25 = output["retrievers"]["bm25"]["categories"]
    dense = output["retrievers"]["dense"]["categories"]
    lex = output.get("lexical_failure_comparison", {})

    def top_cats(stats: dict[str, dict[str, Any]], n: int = 3) -> list[tuple[str, int]]:
        ranked = sorted(
            ((k, v["count"]) for k, v in stats.items() if k != "lexical_failure"),
            key=lambda x: -x[1],
        )
        return ranked[:n]

    h_top = top_cats(hybrid)

    lines = [
        "# Design Motivation from Failure Analysis",
        "",
        "Evidence-based analysis of retrieval failures on QA Dataset V2. "
        "This document motivates SAGE-RAG structural modules — no algorithms proposed.",
        "",
        "## Limitations of Existing Retrieval Methods",
        "",
        "### BM25 (strongest baseline, 70.53% Recall@10)",
        "",
        f"- Primary failure modes: {', '.join(CATEGORY_LABELS[c] + f' ({n})' for c, n in top_cats(bm25))}.",
        "- Lexical matching excels on exact terminology but misses semantic paraphrases "
        f"({lex.get('count', 0)} queries where Dense succeeds and BM25 fails).",
        "- Cross-document and version confusion indicate insufficient document-level disambiguation.",
        "",
        "### Dense (52.44% Recall@10)",
        "",
        f"- Primary failure modes: {', '.join(CATEGORY_LABELS[c] + f' ({n})' for c, n in top_cats(dense))}.",
        "- Embedding similarity retrieves semantically related but structurally wrong clauses.",
        "- Hierarchical and cross-reference failures suggest missing structural constraints.",
        "",
        "### Hybrid RRF (65.85% Recall@10)",
        "",
        f"- Primary failure modes: {', '.join(CATEGORY_LABELS[c] + f' ({n})' for c, n in h_top)}.",
        "- Fusion improves recall over Dense but does not resolve structural failures.",
        "- Persistent categories after RRF indicate gaps that rank fusion alone cannot fix.",
        "",
        "## Failures Addressable by Structural Information",
        "",
        "| Failure Category | Structural Signal Needed |",
        "|------------------|-------------------------|",
        "| Hierarchical | Parent-child-sibling clause tree |",
        "| Cross-reference | Explicit reference edges (Clause/Annex/Table) |",
        "| Appendix | Annex linkage from normative body |",
        "| Table | Table node with caption + row structure |",
        "| Version | Document family + edition metadata |",
        "| Cross-document | Standard ID / scope metadata |",
        "",
        "## Motivation for SAGE-RAG Modules",
        "",
        "### Hierarchical Graph",
        "",
    ]

    hier = hybrid.get("hierarchical_failure", {})
    if hier.get("count", 0):
        sub = hier.get("subtypes", {})
        lines.extend(
            [
                f"- **{hier['count']} failures** ({hier['percentage_of_failures']:.1f}% of Hybrid misses).",
                f"- Subtypes: {sub or 'parent/child/sibling confusion'}.",
                "- Evidence: retriever returns adjacent clause in same chapter instead of gold child/parent.",
                "- A clause hierarchy graph would let retrieval expand or rerank along document structure.",
                "",
            ]
        )
    else:
        lines.append("- Limited hierarchical failures in current sample.\n")

    lines.extend(["### Cross-reference Graph", ""])
    xref = hybrid.get("cross_reference_failure", {})
    if xref.get("count", 0):
        lines.extend(
            [
                f"- **{xref['count']} failures** where gold text cites other clauses/annexes.",
                "- Evidence: answers require following '见条款 X' / 'See Annex A' pointers.",
                "- Cross-reference edges would connect citing clauses to referenced evidence.",
                "",
            ]
        )
    else:
        lines.append("- Cross-reference failures present in secondary signals; primary classification may overlap with semantic/hierarchical.\n")

    lines.extend(["### Table Nodes", ""])
    tbl = hybrid.get("table_failure", {})
    if tbl.get("count", 0):
        lines.extend(
            [
                f"- **{tbl['count']} failures** where gold evidence contains tables.",
                "- Tabular content is poorly represented in flat text chunks.",
                "- Dedicated table nodes preserve row-column semantics and captions.",
                "",
            ]
        )

    lines.extend(["### Appendix Links", ""])
    app = hybrid.get("appendix_failure", {})
    if app.get("count", 0):
        lines.extend(
            [
                f"- **{app['count']} failures** where gold is in appendix/annex content.",
                "- Appendices are semantically distant from normative questions in embedding space.",
                "- Explicit appendix links from referring clauses would bridge this gap.",
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-Retriever Insights",
            "",
            f"- **Lexical failures (Dense✓ BM25✗):** {lex.get('count', 0)} queries — motivates hybrid semantic+lexical pipeline.",
            f"- **Version failures:** {hybrid.get('version_failure', {}).get('count', 0)} — same standard family, wrong edition.",
            f"- **Cross-document failures:** {hybrid.get('cross_document_failure', {}).get('count', 0)} — wrong standard entirely.",
            f"- **Semantic failures (residual):** {hybrid.get('semantic_failure', {}).get('count', 0)} — similar but wrong clause; graph context may disambiguate.",
            "",
            "## Suggested Future SAGE-RAG Modules (Evidence-Based)",
            "",
            "Ranked by Hybrid failure counts:",
            "",
            f"1. **Table Nodes** — {hybrid.get('table_failure', {}).get('count', 0)} failures; tabular gold evidence poorly retrieved.",
            f"2. **Appendix Links** — {hybrid.get('appendix_failure', {}).get('count', 0)} failures; annex content semantically distant.",
            f"3. **Graph-augmented reranking** — {hybrid.get('semantic_failure', {}).get('count', 0)} residual semantic failures after Hybrid.",
            f"4. **Cross-reference Graph** — {hybrid.get('cross_reference_failure', {}).get('count', 0)} failures; multi-hop via explicit references.",
            f"5. **Document/Version Metadata Layer** — {hybrid.get('cross_document_failure', {}).get('count', 0)} cross-doc + {hybrid.get('version_failure', {}).get('count', 0)} version failures.",
            f"6. **Hierarchical Graph** — {hybrid.get('hierarchical_failure', {}).get('count', 0)} failures; parent/child/sibling clause confusion.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_index = load_units(UNITS_PATH)
    qa_index = load_qa_index(QA_PATH)
    total_queries = len(qa_index)

    retriever_records: dict[str, dict[str, dict[str, Any]]] = {}
    failures_by_retriever: dict[str, list[FailureRecord]] = {}

    for name, path in RETRIEVER_PATHS.items():
        records = load_retriever_records(path)
        retriever_records[name] = records
        failures_by_retriever[name] = analyze_retriever(name, records, unit_index, qa_index)

    lexical = detect_lexical_failures(
        retriever_records["dense"],
        retriever_records["bm25"],
    )

    output: dict[str, Any] = {
        "dataset": "qa_pairs_v2",
        "total_queries": total_queries,
        "retrievers": {},
        "lexical_failure_comparison": {
            "count": len(lexical),
            "percentage_of_all_queries": round(100.0 * len(lexical) / total_queries, 2),
            "example_ids": [x["qa_id"] for x in lexical[:8]],
        },
    }

    for name, failures in failures_by_retriever.items():
        cats = build_category_stats(failures, total_queries)
        output["retrievers"][name] = {
            "failure_count": len(failures),
            "failure_rate_pct": round(100.0 * len(failures) / total_queries, 2),
            "categories": cats,
        }

    write_failure_statistics_json(output, output_dir / "failure_statistics.json")

    examples_by_retriever = {
        name: pick_representative_examples(fails, per_category=2)
        for name, fails in failures_by_retriever.items()
    }

    write_failure_statistics_md(output, output_dir / "failure_statistics.md")
    write_failure_examples_md(
        examples_by_retriever,
        unit_index,
        lexical,
        output_dir / "failure_examples.md",
    )

    hybrid_cats = output["retrievers"]["hybrid"]["categories"]
    plot_failure_distribution(
        hybrid_cats,
        len(lexical),
        output_dir / "failure_distribution.png",
    )

    write_design_motivation(output, output_dir / "design_motivation.md")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval failure analysis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for failure analysis outputs",
    )
    args = parser.parse_args()
    result = run_analysis(args.output_dir)
    print(f"Failure analysis complete. Output: {args.output_dir}")
    for name, summary in result["retrievers"].items():
        print(f"  {name}: {summary['failure_count']} failures ({summary['failure_rate_pct']}%)")


if __name__ == "__main__":
    main()
