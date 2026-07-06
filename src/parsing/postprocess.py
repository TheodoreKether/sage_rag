"""Post-processing, title extraction, and quality validation."""

from __future__ import annotations

import re

from .extractors import extract_cover_lines
from .patterns import (
    DOC_TYPE_CN_GB,
    _CLAUSE_LIKE,
    _DOT_LEADER,
    _NOISE_LINE,
    _PAGE_NUMBER_LINE,
    _TITLE_BLACKLIST,
    _TRAILING_PAGE_NUM,
    normalize_unicode,
)
from .structure_builder import Chapter, Clause, DocumentStructure


def _title_is_blacklisted(line: str) -> bool:
    stripped = line.strip()
    for phrase in _TITLE_BLACKLIST:
        if phrase in stripped:
            return True
    if re.match(r"^(?:ICS|ISBN|PDF disclaimer)", stripped, re.I):
        return True
    if re.match(r"^GB/T\s+\d", stripped, re.I):
        return True
    if re.match(r"^ISO(?:/IEC)?\s+\d", stripped, re.I):
        return True
    if re.match(r"^IEC(?:\s+TS)?\s+\d", stripped, re.I):
        return True
    return False


def extract_title(
    pages: list[tuple[int, str]],
    standard_id: str,
    doc_type: str,
) -> str:
    """Extract human-readable document title from cover pages."""
    cover_lines = extract_cover_lines(pages, max_pages=3)
    sid_compact = standard_id.replace(" ", "").replace("/", "").lower()

    cn_candidates: list[str] = []
    en_candidates: list[str] = []

    for line in cover_lines:
        if len(line) < 6 or _title_is_blacklisted(line):
            continue
        if sid_compact and sid_compact[:8] in line.replace(" ", "").replace("/", "").lower():
            continue
        if _NOISE_LINE.match(line) or _PAGE_NUMBER_LINE.match(line):
            continue
        if _CLAUSE_LIKE.match(line):
            continue
        if re.match(r"^(?:发布|实施|代替|Preface|Foreword|目\s*次|Contents|Warning)", line, re.I):
            continue
        if re.search(r"[\u4e00-\u9fff]", line):
            if "标准" in line and len(line) < 20:
                continue
            cn_candidates.append(line)
        elif re.search(r"[A-Za-z]{4,}", line) and len(line) >= 12:
            en_candidates.append(line)

    if doc_type == DOC_TYPE_CN_GB and cn_candidates:
        return max(cn_candidates[:6], key=len)

    for pool in (en_candidates, cn_candidates):
        if pool:
            scored = sorted(
                pool[:10],
                key=lambda s: (sum(c.isalpha() for c in s), len(s)),
                reverse=True,
            )
            return scored[0]

    return ""


def _clause_quality_score(clause: Clause) -> int:
    score = clause.page * 10 + min(len(clause.text), 500)
    if _DOT_LEADER.search(clause.text):
        score -= 500
    if _TRAILING_PAGE_NUM.search(clause.text) and len(clause.text) < 80:
        score -= 300
    if re.search(r"[。；.!?]", clause.text):
        score += 50
    if len(clause.text.strip()) < 8:
        score -= 200
    return score


def _is_garbage_clause(clause: Clause) -> bool:
    text = clause.text.strip()
    if not text:
        return True
    if _DOT_LEADER.search(text) and len(text) < 100:
        return True
    if re.match(r"^.{0,5}$", text):
        return True
    if re.match(r"^(?:I|II|III|IV|V|VI)$", text):
        return True
    return False


def dedupe_clauses(clauses: list[Clause]) -> list[Clause]:
    """Keep the best clause when duplicate clause_id appears in one chapter."""
    best: dict[str, Clause] = {}
    order: list[str] = []
    for clause in clauses:
        if _is_garbage_clause(clause):
            continue
        if clause.clause_id not in best:
            best[clause.clause_id] = clause
            order.append(clause.clause_id)
            continue
        if _clause_quality_score(clause) > _clause_quality_score(best[clause.clause_id]):
            best[clause.clause_id] = clause
    return [best[cid] for cid in order]


def postprocess_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Deduplicate clauses and drop empty chapters."""
    cleaned: list[Chapter] = []
    for chapter in chapters:
        if chapter.chapter_id == "front_matter":
            continue
        chapter.clauses = dedupe_clauses(chapter.clauses)
        if chapter.clauses:
            cleaned.append(chapter)
    return cleaned


def validate_structure(structure: DocumentStructure) -> list[str]:
    """Return human-readable quality warnings."""
    warnings: list[str] = []
    if not structure.title:
        warnings.append("missing title")
    if not structure.chapters:
        warnings.append("no chapters extracted")
        return warnings

    total_clauses = sum(len(ch.clauses) for ch in structure.chapters)
    if total_clauses < 3:
        warnings.append(f"very few clauses ({total_clauses})")

    for chapter in structure.chapters:
        for clause in chapter.clauses:
            root = clause.clause_id.split(".")[0]
            if root.isdigit():
                expected_gb = f"第{root}章"
                if (
                    structure.doc_type == DOC_TYPE_CN_GB
                    and chapter.chapter_id not in (root, expected_gb)
                    and not chapter.chapter_id.startswith("附录")
                ):
                    warnings.append(
                        f"clause {clause.clause_id} under chapter {chapter.chapter_id}"
                    )
                    break

    dup_count = 0
    for chapter in structure.chapters:
        ids = [c.clause_id for c in chapter.clauses]
        dup_count += len(ids) - len(set(ids))
    if dup_count:
        warnings.append(f"{dup_count} duplicate clause_ids remain after dedupe")

    return warnings[:10]


def finalize_structure(structure: DocumentStructure) -> DocumentStructure:
    structure.chapters = postprocess_chapters(structure.chapters)
    warnings = validate_structure(structure)
    structure.quality = {
        **structure.quality,
        "clause_count": sum(len(ch.clauses) for ch in structure.chapters),
        "chapter_count": len(structure.chapters),
        "warnings": warnings,
    }
    return structure
