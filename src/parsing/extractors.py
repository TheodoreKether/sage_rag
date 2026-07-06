"""PDF text extraction and line preprocessing."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .patterns import (
    PatternSet,
    _NOISE_LINE,
    _PAGE_NUMBER_LINE,
    _SENTENCE_END,
    build_patterns,
    is_structural_line,
    is_toc_line,
    normalize_clause_number_line,
    normalize_unicode,
)

logger = logging.getLogger(__name__)


def extract_pages_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []
    try:
        for idx in range(len(doc)):
            page = doc[idx]
            text = page.get_text("text") or ""
            pages.append((idx + 1, text))
    finally:
        doc.close()
    return pages


def extract_pages_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((idx + 1, text))
    return pages


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract per-page text; PyMuPDF first, pdfplumber as fallback."""
    try:
        pages = extract_pages_pymupdf(pdf_path)
        if any(text.strip() for _, text in pages):
            logger.debug("Extracted %d pages via PyMuPDF: %s", len(pages), pdf_path.name)
            return pages
        logger.info("PyMuPDF returned empty text for %s, trying pdfplumber", pdf_path.name)
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s (%s), trying pdfplumber", pdf_path.name, exc)

    try:
        pages = extract_pages_pdfplumber(pdf_path)
        logger.debug("Extracted %d pages via pdfplumber: %s", len(pages), pdf_path.name)
        return pages
    except Exception as exc:
        raise RuntimeError(f"Both extractors failed for {pdf_path.name}: {exc}") from exc


def detect_repeated_lines(pages: list[tuple[int, str]], min_ratio: float = 0.35) -> set[str]:
    """Identify running headers/footers that repeat across pages."""
    page_count = max(len(pages), 1)
    counter: Counter[str] = Counter()
    for _, text in pages:
        seen_on_page: set[str] = set()
        for raw_line in text.splitlines():
            line = normalize_unicode(raw_line).strip()
            if not line or _PAGE_NUMBER_LINE.match(line):
                continue
            if line not in seen_on_page:
                counter[line] += 1
                seen_on_page.add(line)
    threshold = max(3, int(page_count * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


def merge_broken_lines(
    lines: list[tuple[int, str]], patterns: PatternSet
) -> list[tuple[int, str]]:
    """Join wrapped lines that do not start a new structural element."""
    if not lines:
        return []

    merged: list[tuple[int, str]] = []
    for page, line in lines:
        if merged and not is_structural_line(line, patterns):
            prev_page, prev_text = merged[-1]
            if not is_structural_line(prev_text, patterns) and not _SENTENCE_END.search(prev_text):
                merged[-1] = (prev_page, f"{prev_text} {line}")
                continue
        merged.append((page, line))
    return merged


def merge_number_fragments(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Join PDF-broken clause numbers like '7.' + '1.' + '1 状态数据' -> '7.1.1 状态数据'."""
    merged: list[tuple[int, str]] = []
    fragments: list[str] = []
    fragment_page = 0

    def flush_fragments() -> None:
        nonlocal fragments, fragment_page
        for frag in fragments:
            merged.append((fragment_page, frag))
        fragments = []

    for page, line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d+\.?", stripped):
            if not fragments:
                fragment_page = page
            fragments.append(stripped.rstrip("."))
            continue

        if fragments and re.match(r"^(\d+)\s+(\S)", line):
            parts = fragments + [re.match(r"^(\d+)", line).group(1)]
            clause_id = ".".join(parts)
            title = re.sub(r"^\d+\s+", "", line).strip()
            merged.append((page, f"{clause_id} {title}"))
            fragments = []
            continue

        if fragments and re.match(r"^[\u4e00-\u9fffA-Za-z]", line):
            clause_id = ".".join(fragments)
            merged.append((page, f"{clause_id} {line}"))
            fragments = []
            continue

        flush_fragments()
        merged.append((page, line))

    flush_fragments()
    return merged


def preprocess_lines(
    pages: list[tuple[int, str]],
    patterns: PatternSet,
    *,
    body_start_page: int,
    skip_toc_lines: bool = True,
) -> list[tuple[int, str]]:
    """Filter headers/footers, TOC lines, and merge broken lines."""
    repeated = detect_repeated_lines(pages)
    filtered: list[tuple[int, str]] = []
    for page, line in _iter_lines(pages):
        if page < body_start_page:
            continue
        if line in repeated:
            continue
        if _PAGE_NUMBER_LINE.match(line) or _NOISE_LINE.match(line):
            continue
        if skip_toc_lines and is_toc_line(line):
            continue
        line = normalize_clause_number_line(line)
        filtered.append((page, line))
    filtered = merge_number_fragments(filtered)
    return merge_broken_lines(filtered, patterns)


def _iter_lines(pages: list[tuple[int, str]]):
    for page_num, text in pages:
        for raw_line in text.splitlines():
            line = normalize_unicode(raw_line).strip()
            if line:
                yield page_num, line


def extract_cover_lines(pages: list[tuple[int, str]], max_pages: int = 2) -> list[str]:
    lines: list[str] = []
    for page_num, text in pages[:max_pages]:
        if page_num > max_pages:
            break
        for raw_line in text.splitlines():
            line = normalize_unicode(raw_line).strip()
            if line:
                lines.append(line)
    return lines


def extract_metadata_from_cover(pages: list[tuple[int, str]], doc_type: str) -> dict:
    """Pull publish dates, ICS, and English title hints from cover pages."""
    blob = " ".join(extract_cover_lines(pages))
    metadata: dict = {}

    ics = re.search(r"ICS[\s:]*([\d\.\s;L]+)", blob, re.I)
    if ics:
        metadata["ics"] = re.sub(r"\s+", " ", ics.group(1)).strip()

    for label, key in (
        (r"(\d{4}-\d{2}-\d{2})\s*发布", "publish_date"),
        (r"(\d{4}-\d{2}-\d{2})\s*实施", "implement_date"),
        (r"(\d{4}-\d{2}-\d{2})\s*Published", "publish_date"),
    ):
        match = re.search(label, blob, re.I)
        if match and key not in metadata:
            metadata[key] = match.group(1)

    if doc_type in ("ISO", "IEC"):
        edition = re.search(r"(?:Edition|First edition|Third edition)\s+([\d\.\s\w-]+)", blob, re.I)
        if edition:
            metadata["edition"] = edition.group(1).strip()

    return metadata
