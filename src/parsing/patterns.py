"""Regex patterns and text normalization for standards document parsing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DOC_TYPE_CN_GB = "CN_GB"
DOC_TYPE_ISO = "ISO"
DOC_TYPE_IEC = "IEC"
DOC_TYPE_ENTERPRISE = "ENTERPRISE"

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_SENTENCE_END = re.compile(r"[。；：.!?;:…)\]】」』\"'\u3002\uFF1B\uFF1A]$")
_TABLE_REFERENCE = re.compile(
    r"^(?:给出了|见|如下|所列|中|shows|lists|provides)", re.IGNORECASE
)
_CLAUSE_LIKE = re.compile(r"^\d+(?:\.\d+)+\s")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
_NOISE_LINE = re.compile(
    r"^(?:版权所有|All rights reserved|ICS\s+\d|ISBN|GB/T\s+\d|ISO/IEC\s+\d).*$",
    re.IGNORECASE,
)
_TOC_LINE = re.compile(
    r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+.+?\.{3,}\s*\d+\s*$"
)
_TOC_HEADER = re.compile(r"^(?:目\s*次|Contents|Table of contents)$", re.IGNORECASE)
_DOT_LEADER = re.compile(r"\.{4,}")
_TRAILING_PAGE_NUM = re.compile(r"\s+\d{1,4}\s*$")

_TITLE_BLACKLIST = (
    "中华人民共和国国家标准",
    "国家标准化管理委员会",
    "国家市场监督管理总局",
    "Copyright International Organization",
    "International Standard",
    "INTERNATIONAL STANDARD",
    "TECHNICAL SPECIFICATION",
    "Standard",
    "Published in Switzerland",
    "All rights reserved",
)


@dataclass(frozen=True)
class PatternSet:
    gb_chapter: re.Pattern[str]
    iso_chapter: re.Pattern[str]
    top_level: re.Pattern[str]
    clause: re.Pattern[str]
    annex_clause: re.Pattern[str]
    table: re.Pattern[str]
    annex: re.Pattern[str]


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def cn_numeral_to_int(text: str) -> str:
    """Best-effort conversion of simple Chinese chapter numerals to Arabic."""
    text = text.strip()
    if text.isdigit():
        return text
    if text in _CN_DIGITS:
        return str(_CN_DIGITS[text])
    if text.startswith("十"):
        rest = text[1:]
        if not rest:
            return "10"
        return str(10 + _CN_DIGITS.get(rest, 0))
    if "十" in text:
        parts = text.split("十", 1)
        tens = _CN_DIGITS.get(parts[0], 0) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return str(tens * 10 + ones)
    return text


def normalize_clause_number_line(line: str) -> str:
    """Fix spaced numbering such as '7. 1. 2' -> '7.1.2' at line start."""
    match = re.match(r"^(\d+(?:\.\s*\d+)*)", line)
    if not match:
        return line
    prefix = re.sub(r"\.\s+", ".", match.group(1))
    prefix = re.sub(r"\.\s*$", ".", prefix)
    return prefix + line[match.end() :]


def build_patterns(doc_type: str) -> PatternSet:
    """Return compiled regex patterns tuned to the document family."""
    gb_chapter = re.compile(r"^第\s*([0-9一二三四五六七八九十百千]+)\s*章\s*(.*)$")
    iso_chapter = re.compile(r"^(?:Clause|条款)\s+(\d+)\s*(.*)$", re.IGNORECASE)
    top_level = re.compile(r"^(\d{1,2})\s+(?![.\d])(\S.*)$")
    clause = re.compile(r"^(\d+(?:\.\d+)+)\s+(\S.*)$")
    annex_clause = re.compile(r"^([A-Za-z]\.\d+(?:\.\d+)*)\s+(\S.*)$")
    table = re.compile(
        r"^(?:表\s*(\d+(?:\.\d+)*)|Table\s+(\d+(?:\.\d+|[A-Za-z])*(?:\.\d+)*))\s*(.*)$",
        re.IGNORECASE,
    )
    annex = re.compile(
        r"^(?:附录\s*([A-Za-z])|Annex\s+([A-Za-z]))\s*(.*)$",
        re.IGNORECASE,
    )

    if doc_type == DOC_TYPE_CN_GB:
        top_level = re.compile(r"^([1-9]|1[0-9]|20)\s+(?![.\d])([\u4e00-\u9fff\S].*)$")
        clause = re.compile(r"^(\d+(?:\.\d+)+)\s+([\u4e00-\u9fff\S].*)$")
        table = re.compile(r"^表\s*(\d+(?:\.\d+)*)\s*(.*)$")
        annex = re.compile(r"^附录\s*([A-Za-zＡ-Ｚ])\s*(.*)$")

    elif doc_type in (DOC_TYPE_ISO, DOC_TYPE_IEC):
        top_level = re.compile(r"^([1-9]|1[0-9]|20)\s+(?![.\d])([A-Za-z\S].*)$")
        clause = re.compile(r"^(\d+(?:\.\d+)+)\s+([A-Za-z\S].*)$")
        table = re.compile(
            r"^Table\s+(\d+(?:\.\d+|[A-Za-z])*(?:\.\d+)*)\s*(.*)$",
            re.IGNORECASE,
        )
        annex = re.compile(r"^Annex\s+([A-Za-z])\s*(.*)$", re.IGNORECASE)

    return PatternSet(
        gb_chapter=gb_chapter,
        iso_chapter=iso_chapter,
        top_level=top_level,
        clause=clause,
        annex_clause=annex_clause,
        table=table,
        annex=annex,
    )


def is_structural_line(line: str, patterns: PatternSet) -> bool:
    return any(
        p.match(line)
        for p in (
            patterns.gb_chapter,
            patterns.iso_chapter,
            patterns.top_level,
            patterns.clause,
            patterns.annex_clause,
            patterns.table,
            patterns.annex,
        )
    )


def is_toc_line(line: str) -> bool:
    if _TOC_HEADER.match(line.strip()):
        return True
    if _DOT_LEADER.search(line) and _TRAILING_PAGE_NUM.search(line):
        return True
    return bool(_TOC_LINE.match(line))


def is_valid_top_level(line: str, *, in_body: bool, active_chapter_id: str | None = None) -> bool:
    """Reject TOC-style or garbage top-level headings."""
    if not in_body:
        return False
    if _DOT_LEADER.search(line):
        return False
    if _TRAILING_PAGE_NUM.search(line) and len(line) < 100:
        return False
    if len(line) > 120:
        return False
    match = re.match(r"^(\d{1,2})\s+", line)
    if not match:
        return False
    num = int(match.group(1))
    if active_chapter_id and active_chapter_id.isdigit():
        current = int(active_chapter_id)
        if num < current:
            return False
        if num == 1 and current >= 3:
            return False
    title_part = re.sub(r"^\d{1,2}\s+", "", line).strip()
    if len(title_part) < 2:
        return False
    if re.match(r"^(?:min|max|mm|cm|s|kg|kV)\.?$", title_part, re.I):
        return False
    return True
