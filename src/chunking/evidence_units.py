"""Evidence Unit construction from parsed standard documents."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

logger = logging.getLogger(__name__)

MAX_UNIT_TOKENS = 512
WARN_UNIT_TOKENS = 800

_APPENDIX_CHAPTER = re.compile(r"^(?:附录|Annex)\s*[A-Z]?$", re.IGNORECASE)
_TABLE_CLAUSE_ID = re.compile(r"^(?:表\d+|Table\s+\d+)", re.IGNORECASE)
_TABLE_IN_TEXT = re.compile(r"(?:表\s*\d+|Table\s+\d+)", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；.!?;])\s+")
_BLANK_LINE = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _tiktoken_estimate(text: str) -> int | None:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count; prefer tiktoken when installed."""
    if not text:
        return 0
    exact = _tiktoken_estimate(text)
    if exact is not None:
        return exact
    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = max(len(text) - cn_chars, 0)
    return max(1, int(cn_chars / 1.5 + other_chars / 4))


def _chars_per_token(text: str) -> float:
    tokens = estimate_tokens(text)
    if tokens <= 0:
        return 4.0
    return max(len(text) / tokens, 1.0)


# ---------------------------------------------------------------------------
# Semantic splitting
# ---------------------------------------------------------------------------


def _fixed_length_split(text: str, max_tokens: int) -> list[str]:
    cpt = _chars_per_token(text)
    chunk_chars = max(int(max_tokens * cpt), 80)
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        parts.append(text[start:end].strip())
        start = end
    return [p for p in parts if p]


def _merge_segments_under_limit(segments: list[str], max_tokens: int) -> list[str]:
    """Greedy merge of small segments without exceeding token budget."""
    merged: list[str] = []
    buffer = ""
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        candidate = f"{buffer}\n\n{segment}".strip() if buffer else segment
        if estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            if estimate_tokens(segment) <= max_tokens:
                buffer = segment
            else:
                merged.extend(_split_text(segment, max_tokens, depth=1))
                buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _split_by_paragraph(text: str) -> list[str]:
    parts = [p.strip() for p in _BLANK_LINE.split(text) if p.strip()]
    return parts if len(parts) > 1 else []


def _split_by_line(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    return _merge_segments_under_limit(lines, MAX_UNIT_TOKENS)


def _split_by_sentence(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
    if len(parts) <= 1:
        return []
    return _merge_segments_under_limit(parts, MAX_UNIT_TOKENS)


def _split_text(text: str, max_tokens: int, *, depth: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    for splitter in (_split_by_paragraph, _split_by_line, _split_by_sentence):
        parts = splitter(text)
        if len(parts) > 1:
            expanded: list[str] = []
            for part in parts:
                expanded.extend(_split_text(part, max_tokens, depth=depth + 1))
            if len(expanded) > 1:
                return expanded

    return _fixed_length_split(text, max_tokens)


def split_clause_text(text: str, max_tokens: int = MAX_UNIT_TOKENS) -> list[str]:
    """Adaptively split long clause text into semantic segments."""
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    return _split_text(text, max_tokens)


# ---------------------------------------------------------------------------
# Evidence unit model & construction
# ---------------------------------------------------------------------------


def slugify_document_id(standard_id: str, filename: str) -> str:
    base = standard_id.strip() if standard_id else Path(filename).stem
    slug = re.sub(r"[^\w\-./]+", "_", base)
    slug = slug.replace("/", "_").replace(" ", "_")
    return slug or Path(filename).stem


def is_appendix_chapter(chapter_id: str) -> bool:
    return bool(_APPENDIX_CHAPTER.match(chapter_id.strip()))


def contains_table(clause_id: str, text: str) -> bool:
    if _TABLE_CLAUSE_ID.match(clause_id.strip()):
        return True
    return bool(_TABLE_IN_TEXT.search(text))


def make_unit_id(document_id: str, chapter_id: str, parent_clause: str, split_index: int) -> str:
    safe_ch = re.sub(r"[^\w.\-]+", "_", chapter_id)
    safe_cl = re.sub(r"[^\w.\-]+", "_", parent_clause)
    return f"{document_id}::{safe_ch}::{safe_cl}::{split_index}"


@dataclass
class EvidenceUnit:
    unit_id: str
    document_id: str
    document_type: str
    title: str
    chapter_id: str
    chapter_title: str
    parent_clause: str
    page: int
    text: str
    token_length: int
    char_length: int
    split_index: int
    split_total: int
    metadata: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "title": self.title,
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "parent_clause": self.parent_clause,
            "page": self.page,
            "text": self.text,
            "token_length": self.token_length,
            "char_length": self.char_length,
            "split_index": self.split_index,
            "split_total": self.split_total,
            "metadata": self.metadata,
        }


def build_units_from_document(data: dict[str, Any], filename: str) -> list[EvidenceUnit]:
    document_id = slugify_document_id(data.get("standard_id", ""), filename)
    doc_type = data.get("doc_type", "")
    title = data.get("title", "")
    units: list[EvidenceUnit] = []

    for chapter in data.get("chapters", []):
        chapter_id = chapter.get("chapter_id", "")
        chapter_title = chapter.get("chapter_title", "")
        is_appendix = is_appendix_chapter(chapter_id)

        for clause in chapter.get("clauses", []):
            parent_clause = clause.get("clause_id", "")
            text = (clause.get("text") or "").strip()
            page = int(clause.get("page") or chapter.get("page") or 0)
            if not text:
                continue

            segments = split_clause_text(text)
            split_total = len(segments)
            table_flag = contains_table(parent_clause, text)

            for idx, segment in enumerate(segments, start=1):
                units.append(
                    EvidenceUnit(
                        unit_id=make_unit_id(document_id, chapter_id, parent_clause, idx),
                        document_id=document_id,
                        document_type=doc_type,
                        title=title,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        parent_clause=parent_clause,
                        page=page,
                        text=segment,
                        token_length=estimate_tokens(segment),
                        char_length=len(segment),
                        split_index=idx,
                        split_total=split_total,
                        metadata={
                            "contains_table": table_flag,
                            "contains_appendix": is_appendix,
                        },
                    )
                )
    return units


# ---------------------------------------------------------------------------
# I/O, quality, statistics
# ---------------------------------------------------------------------------


def load_parsed_documents(input_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(input_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                yield path, json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path.name, exc)


def inspect_quality(units: list[EvidenceUnit]) -> list[str]:
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for unit in units:
        if not unit.text.strip():
            warnings.append(f"empty evidence: {unit.unit_id}")
        if not unit.parent_clause:
            warnings.append(f"missing parent_clause: {unit.unit_id}")
        if unit.unit_id in seen_ids:
            warnings.append(f"duplicate unit_id: {unit.unit_id}")
        seen_ids.add(unit.unit_id)
        if unit.token_length > WARN_UNIT_TOKENS:
            warnings.append(
                f"token length > {WARN_UNIT_TOKENS} after split: "
                f"{unit.unit_id} ({unit.token_length} tokens)"
            )
    return warnings


@dataclass
class BuildResult:
    units: list[EvidenceUnit]
    warnings: list[str]
    source_files: int


def build_evidence_units(input_dir: Path) -> BuildResult:
    all_units: list[EvidenceUnit] = []
    paths = sorted(Path(input_dir).glob("*.json"))

    for path in tqdm(paths, desc="Building evidence units", unit="file"):
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            all_units.extend(build_units_from_document(data, path.name))
        except Exception as exc:
            logger.warning("Skipped %s: %s", path.name, exc)

    warnings = inspect_quality(all_units)
    return BuildResult(units=all_units, warnings=warnings, source_files=len(paths))


def write_jsonl(units: list[EvidenceUnit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for unit in units:
            fh.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d evidence units to %s", len(units), output_path)


def render_statistics_markdown(units: list[EvidenceUnit], warnings: list[str]) -> str:
    if not units:
        return "# Evidence Unit Statistics\n\nNo evidence units generated.\n"

    token_lengths = [u.token_length for u in units]
    char_lengths = [u.char_length for u in units]
    split_counts = Counter(u.split_total for u in units)
    max_split = max(u.split_total for u in units)
    longest = sorted(units, key=lambda u: u.token_length, reverse=True)[:10]

    lines = [
        "# Evidence Unit Statistics",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Evidence units | {len(units)} |",
        f"| Average token length | {mean(token_lengths):.1f} |",
        f"| Average character length | {mean(char_lengths):.1f} |",
        f"| Maximum token length | {max(token_lengths)} |",
        f"| Maximum split count | {max_split} |",
        "",
        "## Split Count Distribution",
        "",
        "| split_total | Count |",
        "|------------:|------:|",
    ]
    for split_total in sorted(split_counts):
        lines.append(f"| {split_total} | {split_counts[split_total]} |")

    lines.extend(["", "## Top 10 Longest Evidence Units", ""])
    lines.append("| unit_id | parent_clause | token_length | char_length | split |")
    lines.append("|---------|---------------|-------------:|------------:|------:|")
    for unit in longest:
        lines.append(
            f"| `{unit.unit_id}` | `{unit.parent_clause}` | "
            f"{unit.token_length} | {unit.char_length} | "
            f"{unit.split_index}/{unit.split_total} |"
        )

    lines.extend(["", "## Quality Warnings", ""])
    if not warnings:
        lines.append("No quality warnings.")
    else:
        lines.append(f"Total warnings: **{len(warnings)}**")
        lines.append("")
        for warning in warnings[:100]:
            lines.append(f"- {warning}")
        if len(warnings) > 100:
            lines.append(f"- ... and {len(warnings) - 100} more")

    return "\n".join(lines) + "\n"


def run_builder(
    input_dir: Path,
    output_dir: Path,
    *,
    stats_path: Path | None = None,
) -> BuildResult:
    result = build_evidence_units(input_dir)
    jsonl_path = output_dir / "evidence_units.jsonl"
    write_jsonl(result.units, jsonl_path)

    stats_path = stats_path or Path("results/benchmark/evidence_statistics.md")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        render_statistics_markdown(result.units, result.warnings),
        encoding="utf-8",
    )
    logger.info("Wrote statistics to %s", stats_path)

    if result.warnings:
        logger.warning("%d quality warnings (see %s)", len(result.warnings), stats_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build standardized Evidence Units from parsed JSON.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/parsed_json",
        help="Directory with parsed JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/evidence_units",
        help="Output directory (writes evidence_units.jsonl)",
    )
    parser.add_argument(
        "--stats",
        default="results/benchmark/evidence_statistics.md",
        help="Markdown statistics report path",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1

    result = run_builder(
        input_dir,
        Path(args.output),
        stats_path=Path(args.stats),
    )
    logger.info(
        "Built %d evidence units from %d documents",
        len(result.units),
        result.source_files,
    )
    return 0


if __name__ == "__main__":
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
