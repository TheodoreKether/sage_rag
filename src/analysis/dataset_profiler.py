"""Profile parsed standard-document JSON for chunking-strategy research.

Analyzes corpus-level and per-document statistics, quality warnings, and
generates markdown / JSON reports plus matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

logger = logging.getLogger(__name__)

DOC_TYPE_GB = "CN_GB"
DOC_TYPE_ISO = "ISO"
DOC_TYPE_IEC = "IEC"
DOC_TYPE_ENTERPRISE = "ENTERPRISE"

MIN_NORMAL_CLAUSE_LEN = 20
MAX_NORMAL_CLAUSE_LEN = 5000

_APPENDIX_CHAPTER = re.compile(
    r"^(?:附录|Annex)\s*[A-Z]?$|^(?:附录|Annex)\s*[A-Z]$",
    re.IGNORECASE,
)
_TABLE_CLAUSE = re.compile(r"^(?:表\d+|Table\s+\d+)", re.IGNORECASE)

# Normalized numbering pattern buckets for research comparison.
_PATTERN_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("annex_subclause", re.compile(r"^[A-Z]\.\d+(?:\.\d+)*$")),
    ("deep_numeric", re.compile(r"^\d+\.\d+\.\d+(?:\.\d+)*$")),
    ("subclause", re.compile(r"^\d+\.\d+$")),
    ("top_level", re.compile(r"^\d+$")),
    ("gb_table", re.compile(r"^表\d+(?:\.\d+)*$")),
    ("iso_table", re.compile(r"^Table\s+\d+", re.IGNORECASE)),
    ("appendix_chapter", re.compile(r"^(?:附录|Annex)\s*[A-Z]$", re.IGNORECASE)),
    ("gb_chapter", re.compile(r"^第\d+章$")),
    ("other", re.compile(r".+")),
]


@dataclass
class ClauseRecord:
    clause_id: str
    text: str
    page: int
    chapter_id: str
    length: int
    numbering_pattern: str
    is_table: bool
    is_appendix_chapter: bool


@dataclass
class DocumentProfile:
    filename: str
    standard_id: str
    doc_type: str
    title: str
    pages: int
    chapters: int
    clauses: int
    appendix_sections: int
    tables: int
    avg_clause_length: float
    median_clause_length: float
    min_clause_length: int
    max_clause_length: int
    clause_lengths: list[int] = field(default_factory=list)
    clause_patterns: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CorpusProfile:
    document_count: int
    gb_count: int
    iso_count: int
    iec_count: int
    enterprise_count: int
    avg_pages: float
    avg_chapters: float
    avg_clauses: float
    avg_clause_length: float
    max_clause_length: int
    min_clause_length: int
    total_appendix_sections: int
    total_tables: int
    clause_length_distribution: dict[str, int]
    top_clause_patterns: list[tuple[str, int]]
    documents: list[DocumentProfile]
    quality_warnings: list[dict[str, Any]]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def collect_json_paths(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        logger.warning("No JSON files found in %s", input_dir)
    return paths


def load_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def infer_page_count(data: dict[str, Any]) -> int:
    pages: set[int] = set()
    for chapter in data.get("chapters", []):
        if chapter.get("page"):
            pages.add(int(chapter["page"]))
        for clause in chapter.get("clauses", []):
            if clause.get("page"):
                pages.add(int(clause["page"]))
    for entry in data.get("toc", []):
        if entry.get("page"):
            pages.add(int(entry["page"]))
    return max(pages) if pages else 0


def classify_numbering_pattern(clause_id: str) -> str:
    for name, pattern in _PATTERN_RULES:
        if name == "other":
            continue
        if pattern.match(clause_id):
            return name
    return "other"


def is_appendix_chapter(chapter_id: str) -> bool:
    return bool(_APPENDIX_CHAPTER.match(chapter_id.strip()))


def is_table_clause(clause_id: str) -> bool:
    return bool(_TABLE_CLAUSE.match(clause_id.strip()))


def iter_clauses(data: dict[str, Any]) -> list[ClauseRecord]:
    records: list[ClauseRecord] = []
    for chapter in data.get("chapters", []):
        chapter_id = chapter.get("chapter_id", "")
        appendix_ch = is_appendix_chapter(chapter_id)
        for clause in chapter.get("clauses", []):
            text = (clause.get("text") or "").strip()
            clause_id = clause.get("clause_id", "")
            records.append(
                ClauseRecord(
                    clause_id=clause_id,
                    text=text,
                    page=int(clause.get("page") or 0),
                    chapter_id=chapter_id,
                    length=len(text),
                    numbering_pattern=classify_numbering_pattern(clause_id),
                    is_table=is_table_clause(clause_id),
                    is_appendix_chapter=appendix_ch,
                )
            )
    return records


def inspect_document_quality(
    data: dict[str, Any],
    filename: str,
    clauses: list[ClauseRecord],
) -> list[str]:
    warnings: list[str] = []
    chapters = data.get("chapters", [])

    if not chapters:
        warnings.append("no chapters")
    if not clauses:
        warnings.append("no clauses")

    for chapter in chapters:
        cid = chapter.get("chapter_id", "?")
        if not chapter.get("clauses"):
            warnings.append(f"chapter '{cid}' has zero clauses")

    seen_global: set[tuple[str, str]] = set()
    for chapter in chapters:
        chapter_id = chapter.get("chapter_id", "")
        ids_in_chapter = [c.get("clause_id", "") for c in chapter.get("clauses", [])]
        if len(ids_in_chapter) != len(set(ids_in_chapter)):
            dup = [k for k, v in Counter(ids_in_chapter).items() if v > 1]
            warnings.append(
                f"duplicate clause ids in chapter '{chapter_id}': {dup[:5]}"
            )
        for clause_id in ids_in_chapter:
            key = (chapter_id, clause_id)
            if key in seen_global:
                warnings.append(f"duplicate clause id '{clause_id}' in chapter '{chapter_id}'")
            seen_global.add(key)

    for clause in clauses:
        if not clause.text:
            warnings.append(f"empty clause text: {clause.clause_id} (ch {clause.chapter_id})")
        elif clause.length < MIN_NORMAL_CLAUSE_LEN:
            warnings.append(
                f"short clause (<{MIN_NORMAL_CLAUSE_LEN} chars): "
                f"{clause.clause_id} ({clause.length} chars)"
            )
        elif clause.length > MAX_NORMAL_CLAUSE_LEN:
            warnings.append(
                f"long clause (>{MAX_NORMAL_CLAUSE_LEN} chars): "
                f"{clause.clause_id} ({clause.length} chars)"
            )

    if len(chapters) == 1 and len(clauses) <= 2:
        warnings.append("very sparse structure (<=2 clauses)")

    return warnings


def profile_document(path: Path, data: dict[str, Any] | None = None) -> DocumentProfile:
    data = data or load_document(path)
    clauses = iter_clauses(data)
    lengths = [c.length for c in clauses if c.text]
    chapters = data.get("chapters", [])

    appendix_sections = sum(
        1 for ch in chapters if is_appendix_chapter(ch.get("chapter_id", ""))
    )
    tables = sum(1 for c in clauses if c.is_table)
    pattern_counter = Counter(c.numbering_pattern for c in clauses)

    warnings = inspect_document_quality(data, path.name, clauses)

    return DocumentProfile(
        filename=path.name,
        standard_id=data.get("standard_id", ""),
        doc_type=data.get("doc_type", "UNKNOWN"),
        title=data.get("title", ""),
        pages=infer_page_count(data),
        chapters=len(chapters),
        clauses=len(clauses),
        appendix_sections=appendix_sections,
        tables=tables,
        avg_clause_length=mean(lengths) if lengths else 0.0,
        median_clause_length=median(lengths) if lengths else 0.0,
        min_clause_length=min(lengths) if lengths else 0,
        max_clause_length=max(lengths) if lengths else 0,
        clause_lengths=lengths,
        clause_patterns=pattern_counter,
        warnings=warnings,
    )


def bucket_clause_lengths(lengths: list[int]) -> dict[str, int]:
    buckets = {
        "0-49": 0,
        "50-199": 0,
        "200-499": 0,
        "500-999": 0,
        "1000-1999": 0,
        "2000-4999": 0,
        "5000+": 0,
    }
    for length in lengths:
        if length < 50:
            buckets["0-49"] += 1
        elif length < 200:
            buckets["50-199"] += 1
        elif length < 500:
            buckets["200-499"] += 1
        elif length < 1000:
            buckets["500-999"] += 1
        elif length < 2000:
            buckets["1000-1999"] += 1
        elif length < 5000:
            buckets["2000-4999"] += 1
        else:
            buckets["5000+"] += 1
    return buckets


def aggregate_corpus(documents: list[DocumentProfile]) -> CorpusProfile:
    all_lengths = [length for doc in documents for length in doc.clause_lengths]
    global_patterns: Counter = Counter()
    for doc in documents:
        global_patterns.update(doc.clause_patterns)

    quality_warnings = [
        {
            "filename": doc.filename,
            "standard_id": doc.standard_id,
            "warnings": doc.warnings,
        }
        for doc in documents
        if doc.warnings
    ]

    type_counts = Counter(doc.doc_type for doc in documents)

    return CorpusProfile(
        document_count=len(documents),
        gb_count=type_counts.get(DOC_TYPE_GB, 0),
        iso_count=type_counts.get(DOC_TYPE_ISO, 0),
        iec_count=type_counts.get(DOC_TYPE_IEC, 0),
        enterprise_count=type_counts.get(DOC_TYPE_ENTERPRISE, 0),
        avg_pages=mean([d.pages for d in documents]) if documents else 0.0,
        avg_chapters=mean([d.chapters for d in documents]) if documents else 0.0,
        avg_clauses=mean([d.clauses for d in documents]) if documents else 0.0,
        avg_clause_length=mean(all_lengths) if all_lengths else 0.0,
        max_clause_length=max(all_lengths) if all_lengths else 0,
        min_clause_length=min(all_lengths) if all_lengths else 0,
        total_appendix_sections=sum(d.appendix_sections for d in documents),
        total_tables=sum(d.tables for d in documents),
        clause_length_distribution=bucket_clause_lengths(all_lengths),
        top_clause_patterns=global_patterns.most_common(20),
        documents=documents,
        quality_warnings=quality_warnings,
    )


def profile_dataset(input_dir: Path) -> CorpusProfile:
    paths = collect_json_paths(input_dir)
    documents: list[DocumentProfile] = []

    for path in tqdm(paths, desc="Profiling documents", unit="file"):
        try:
            data = load_document(path)
            documents.append(profile_document(path, data))
        except Exception as exc:
            logger.warning("Skipped %s: %s", path.name, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            documents.append(
                DocumentProfile(
                    filename=path.name,
                    standard_id="",
                    doc_type="ERROR",
                    title="",
                    pages=0,
                    chapters=0,
                    clauses=0,
                    appendix_sections=0,
                    tables=0,
                    avg_clause_length=0.0,
                    median_clause_length=0.0,
                    min_clause_length=0,
                    max_clause_length=0,
                    warnings=[f"failed to load: {exc}"],
                )
            )

    return aggregate_corpus(documents)


def corpus_to_json(corpus: CorpusProfile) -> dict[str, Any]:
    return {
        "generated_at": corpus.generated_at,
        "corpus_summary": {
            "document_count": corpus.document_count,
            "gb_count": corpus.gb_count,
            "iso_count": corpus.iso_count,
            "iec_count": corpus.iec_count,
            "enterprise_count": corpus.enterprise_count,
            "avg_pages": round(corpus.avg_pages, 2),
            "avg_chapters": round(corpus.avg_chapters, 2),
            "avg_clauses": round(corpus.avg_clauses, 2),
            "avg_clause_length": round(corpus.avg_clause_length, 2),
            "max_clause_length": corpus.max_clause_length,
            "min_clause_length": corpus.min_clause_length,
            "total_appendix_sections": corpus.total_appendix_sections,
            "total_tables": corpus.total_tables,
            "clause_length_distribution": corpus.clause_length_distribution,
            "top_clause_numbering_patterns": [
                {"pattern": name, "count": count}
                for name, count in corpus.top_clause_patterns
            ],
        },
        "documents": [
            {
                "filename": doc.filename,
                "standard_id": doc.standard_id,
                "doc_type": doc.doc_type,
                "title": doc.title,
                "pages": doc.pages,
                "chapters": doc.chapters,
                "clauses": doc.clauses,
                "appendix_sections": doc.appendix_sections,
                "tables": doc.tables,
                "avg_clause_length": round(doc.avg_clause_length, 2),
                "median_clause_length": round(doc.median_clause_length, 2),
                "min_clause_length": doc.min_clause_length,
                "max_clause_length": doc.max_clause_length,
            }
            for doc in corpus.documents
        ],
        "quality_warnings": corpus.quality_warnings,
    }


def render_markdown_report(corpus: CorpusProfile) -> str:
    lines: list[str] = [
        "# Sage RAG — Parsed Dataset Profile",
        "",
        f"Generated: {corpus.generated_at}",
        "",
        "This report summarizes parsed standard documents under `data/parsed_json/` "
        "to inform chunking and structure-aware retrieval design.",
        "",
        "## Corpus Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Documents | {corpus.document_count} |",
        f"| GB (CN_GB) | {corpus.gb_count} |",
        f"| ISO | {corpus.iso_count} |",
        f"| IEC | {corpus.iec_count} |",
        f"| Enterprise | {corpus.enterprise_count} |",
        f"| Avg pages (inferred) | {corpus.avg_pages:.1f} |",
        f"| Avg chapters | {corpus.avg_chapters:.1f} |",
        f"| Avg clauses | {corpus.avg_clauses:.1f} |",
        f"| Avg clause length (chars) | {corpus.avg_clause_length:.1f} |",
        f"| Min clause length | {corpus.min_clause_length} |",
        f"| Max clause length | {corpus.max_clause_length} |",
        f"| Appendix sections | {corpus.total_appendix_sections} |",
        f"| Detected tables | {corpus.total_tables} |",
        "",
        "## Clause Length Distribution",
        "",
        "| Bucket (chars) | Count |",
        "|----------------|------:|",
    ]
    for bucket, count in corpus.clause_length_distribution.items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Top 20 Clause Numbering Patterns",
            "",
            "| Pattern | Count |",
            "|---------|------:|",
        ]
    )
    for name, count in corpus.top_clause_patterns:
        lines.append(f"| `{name}` | {count} |")

    lines.extend(
        [
            "",
            "## Per-Document Statistics",
            "",
            "| File | Standard | Type | Pages | Chapters | Clauses | Avg Len | Max Len |",
            "|------|----------|------|------:|---------:|--------:|--------:|--------:|",
        ]
    )
    for doc in corpus.documents:
        lines.append(
            f"| {doc.filename} | {doc.standard_id} | {doc.doc_type} | "
            f"{doc.pages} | {doc.chapters} | {doc.clauses} | "
            f"{doc.avg_clause_length:.0f} | {doc.max_clause_length} |"
        )

    lines.extend(["", "## Quality Warnings", ""])
    if not corpus.quality_warnings:
        lines.append("No quality warnings detected.")
    else:
        for item in corpus.quality_warnings:
            lines.append(f"### `{item['filename']}` ({item['standard_id']})")
            for warning in item["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

    lines.extend(
        [
            "## Research Implications",
            "",
            "### 1. Retrieval unit",
            "",
            _retrieval_unit_notes(corpus),
            "",
            "### 2. Chunking granularity",
            "",
            _chunking_notes(corpus),
            "",
            "### 3. Hierarchical retrieval",
            "",
            _hierarchy_notes(corpus),
            "",
            "### 4. Graph construction feasibility",
            "",
            _graph_notes(corpus),
            "",
            "## Figures",
            "",
            "- `results/figures/clause_length_histogram.png`",
            "- `results/figures/clauses_per_document.png`",
            "- `results/figures/chapter_count_distribution.png`",
            "",
        ]
    )
    return "\n".join(lines)


def _retrieval_unit_notes(corpus: CorpusProfile) -> str:
    median_len = median([length for d in corpus.documents for length in d.clause_lengths]) if corpus.documents else 0
    if corpus.avg_clause_length > 800:
        return (
            f"Median inferred clause length is ~{median_len:.0f} characters and the maximum "
            f"is {corpus.max_clause_length}. Many clauses are long enough to serve as primary "
            "retrieval units, but sub-clause splitting may still be needed for ISO deep "
            "numbering (e.g. 6.2.2.1)."
        )
    return (
        "Clauses are moderately sized on average; clause-level retrieval units are a "
        "reasonable default, with optional merging of very short clauses."
    )


def _chunking_notes(corpus: CorpusProfile) -> str:
    deep = sum(count for name, count in corpus.top_clause_patterns if name == "deep_numeric")
    sub = sum(count for name, count in corpus.top_clause_patterns if name == "subclause")
    return (
        f"The corpus contains {sub} sub-clauses (`X.Y`) and {deep} deep sub-clauses "
        f"(`X.Y.Z+`). Chunking should preserve clause IDs and chapter context. "
        "For GB documents, top-level sections often map 1:1 to chapters; ISO/IEC "
        "documents benefit from sub-clause granularity."
    )


def _hierarchy_notes(corpus: CorpusProfile) -> str:
    return (
        f"Average {corpus.avg_chapters:.1f} chapters and {corpus.avg_clauses:.1f} clauses "
        f"per document across {corpus.document_count} standards. Hierarchical retrieval "
        "(chapter → clause) is suitable when chapter titles and clause numbering are "
        "consistent; heterogeneous numbering schemes across GB / ISO / IEC suggest "
        "doc-type-specific routing."
    )


def _graph_notes(corpus: CorpusProfile) -> str:
    return (
        f"Detected {corpus.total_tables} table clauses and "
        f"{corpus.total_appendix_sections} appendix sections. Cross-reference edges "
        "(citations, 'see Annex A', 'Table 1') can be extracted from clause text in a "
        "later pass. Graph construction is feasible at chapter/clause/table granularity."
    )


def generate_figures(corpus: CorpusProfile, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    all_lengths = [length for doc in corpus.documents for length in doc.clause_lengths]
    if all_lengths:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_lengths, bins=30, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Clause length (characters)")
        ax.set_ylabel("Frequency")
        ax.set_title("Clause Length Distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "clause_length_histogram.png", dpi=150)
        plt.close(fig)

    doc_names = [doc.filename.replace(".json", "")[:20] for doc in corpus.documents]
    clause_counts = [doc.clauses for doc in corpus.documents]
    if clause_counts:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(clause_counts)), clause_counts, color="#55A868")
        ax.set_xticks(range(len(doc_names)))
        ax.set_xticklabels(doc_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Clause count")
        ax.set_title("Clauses per Document")
        fig.tight_layout()
        fig.savefig(figures_dir / "clauses_per_document.png", dpi=150)
        plt.close(fig)

    chapter_counts = [doc.chapters for doc in corpus.documents]
    if chapter_counts:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(chapter_counts, bins=range(min(chapter_counts), max(chapter_counts) + 2),
                color="#C44E52", edgecolor="white", align="left")
        ax.set_xlabel("Chapters per document")
        ax.set_ylabel("Frequency")
        ax.set_title("Chapter Count Distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "chapter_count_distribution.png", dpi=150)
        plt.close(fig)

    logger.info("Saved figures to %s", figures_dir)


def save_reports(
    corpus: CorpusProfile,
    *,
    markdown_path: Path,
    json_path: Path,
    figures_dir: Path,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_path.write_text(render_markdown_report(corpus), encoding="utf-8")
    json_path.write_text(
        json.dumps(corpus_to_json(corpus), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generate_figures(corpus, figures_dir)
    logger.info("Wrote %s", markdown_path)
    logger.info("Wrote %s", json_path)


def run_profiler(
    input_dir: Path,
    *,
    markdown_path: Path,
    json_path: Path,
    figures_dir: Path,
) -> CorpusProfile:
    corpus = profile_dataset(input_dir)
    save_reports(
        corpus,
        markdown_path=markdown_path,
        json_path=json_path,
        figures_dir=figures_dir,
    )
    return corpus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile parsed standard-document JSON corpus.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/parsed_json",
        help="Directory containing parsed JSON files",
    )
    parser.add_argument(
        "--markdown",
        default="results/dataset_profile.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--json",
        default="results/dataset_statistics.json",
        help="Output JSON statistics path",
    )
    parser.add_argument(
        "--figures",
        default="results/figures",
        help="Output directory for charts",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        corpus = run_profiler(
            Path(args.input),
            markdown_path=Path(args.markdown),
            json_path=Path(args.json),
            figures_dir=Path(args.figures),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Profiled %d documents, %d quality-flagged files",
        corpus.document_count,
        len(corpus.quality_warnings),
    )
    return 0


if __name__ == "__main__":
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
