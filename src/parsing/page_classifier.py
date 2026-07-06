"""Page role classification and body-start detection."""

from __future__ import annotations

import re

from .patterns import (
    DOC_TYPE_CN_GB,
    DOC_TYPE_IEC,
    DOC_TYPE_ISO,
    _TOC_HEADER,
    build_patterns,
    is_toc_line,
    normalize_unicode,
)


def find_body_start_page(pages: list[tuple[int, str]], doc_type: str) -> int:
    """Locate the first page that contains real body content (not TOC/cover)."""
    patterns = build_patterns(doc_type)
    best = len(pages) + 1

    for page_num, text in pages:
        lines = [normalize_unicode(ln).strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        for line in lines:
            if patterns.gb_chapter.match(line):
                return page_num
            if patterns.iso_chapter.match(line):
                return page_num
            if patterns.annex.match(line) and page_num > 2:
                return min(best, page_num)

        toc_like = sum(1 for ln in lines if is_toc_line(ln) or _TOC_HEADER.match(ln))
        if toc_like >= 3:
            continue

        for line in lines:
            if patterns.clause.match(line) and not is_toc_line(line):
                return min(best, page_num)
            if doc_type == DOC_TYPE_CN_GB:
                if re.match(r"^1\.1\s+[\u4e00-\u9fff]", line) and not is_toc_line(line):
                    return min(best, page_num)
            if doc_type in (DOC_TYPE_ISO, DOC_TYPE_IEC):
                if re.match(r"^1(?:\.\d+)+\s+[A-Za-z]", line) and not is_toc_line(line):
                    return min(best, page_num)
                if re.match(r"^1\s+(?:Scope|General|Introduction)\b", line, re.I):
                    return min(best, page_num)

        if doc_type == DOC_TYPE_CN_GB:
            for line in lines:
                if re.match(r"^[1-9]\s+范围\s*$", line) and page_num >= 4:
                    return min(best, page_num)

    if best <= len(pages):
        return best
    return 1


def extract_toc_entries(pages: list[tuple[int, str]], body_start_page: int) -> list[dict]:
    """Parse table-of-contents lines from pre-body pages."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for page_num, text in pages:
        if page_num >= body_start_page:
            break
        for raw_line in text.splitlines():
            line = normalize_unicode(raw_line).strip()
            if not is_toc_line(line):
                continue
            match = re.match(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+(.+?)\.{3,}\s*(\d+)\s*$", line)
            if not match:
                continue
            clause_id, title, target_page = match.groups()
            title = re.sub(r"\.+$", "", title).strip()
            key = (clause_id, title)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "clause_id": clause_id,
                    "title": title,
                    "page": int(target_page),
                }
            )
    return entries
