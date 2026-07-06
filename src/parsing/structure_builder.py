"""Incremental builder for chapter / clause hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .patterns import (
    DOC_TYPE_CN_GB,
    DOC_TYPE_IEC,
    DOC_TYPE_ISO,
    _TABLE_REFERENCE,
    build_patterns,
    cn_numeral_to_int,
    is_valid_top_level,
)


@dataclass
class Clause:
    clause_id: str
    text: str
    page: int


@dataclass
class Chapter:
    chapter_id: str
    chapter_title: str
    page: int = 0
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class DocumentStructure:
    standard_id: str
    doc_type: str
    title: str
    metadata: dict = field(default_factory=dict)
    toc: list[dict] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "standard_id": self.standard_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "metadata": self.metadata,
            "toc": self.toc,
            "chapters": [
                {
                    "chapter_id": ch.chapter_id,
                    "chapter_title": ch.chapter_title,
                    "page": ch.page,
                    "clauses": [
                        {
                            "clause_id": cl.clause_id,
                            "text": cl.text.strip(),
                            "page": cl.page,
                        }
                        for cl in ch.clauses
                        if cl.text.strip()
                    ],
                }
                for ch in self.chapters
                if ch.clauses
            ],
            "quality": self.quality,
        }


@dataclass
class _ActiveClause:
    clause_id: str
    page: int
    title_part: str = ""
    body_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = [self.title_part] if self.title_part else []
        parts.extend(self.body_lines)
        return " ".join(p.strip() for p in parts if p.strip())


class StructureBuilder:
    """Build chapter/clause tree from pre-filtered body lines."""

    def __init__(self, doc_type: str) -> None:
        self.doc_type = doc_type
        self.patterns = build_patterns(doc_type)
        self.chapters: dict[str, Chapter] = {}
        self.chapter_order: list[str] = []
        self._chapter_aliases: dict[str, str] = {}
        self._active_chapter_id: str | None = None
        self._active_clause: _ActiveClause | None = None

    def _ensure_chapter(self, chapter_id: str, title: str = "", page: int = 0) -> Chapter:
        if chapter_id not in self.chapters:
            chapter = Chapter(
                chapter_id=chapter_id,
                chapter_title=title.strip(),
                page=page,
            )
            self.chapters[chapter_id] = chapter
            self.chapter_order.append(chapter_id)
        else:
            chapter = self.chapters[chapter_id]
            if title.strip() and not chapter.chapter_title:
                chapter.chapter_title = title.strip()
            if page and not chapter.page:
                chapter.page = page
        return chapter

    def _resolve_chapter_id(self, key: str) -> str:
        if key in self._chapter_aliases:
            return self._chapter_aliases[key]
        chapter_key = f"第{key}章"
        if self.doc_type == DOC_TYPE_CN_GB and chapter_key in self.chapters:
            return chapter_key
        annex_key = f"附录{key}" if self.doc_type == DOC_TYPE_CN_GB else f"Annex {key}"
        if len(key) == 1 and key.isalpha() and annex_key in self.chapters:
            return annex_key
        return key

    def _chapter_key_from_clause(self, clause_id: str) -> str:
        if clause_id.startswith("表") or clause_id.lower().startswith("table"):
            return self._active_chapter_id or "tables"
        if re.match(r"^[A-Za-z](?:\.\d+)+", clause_id):
            letter = clause_id[0].upper()
            annex_id = f"附录{letter}" if self.doc_type == DOC_TYPE_CN_GB else f"Annex {letter}"
            return annex_id if annex_id in self.chapters else self._resolve_chapter_id(letter)
        root = clause_id.split(".")[0]
        return self._resolve_chapter_id(root)

    def _flush_clause(self) -> None:
        if not self._active_clause:
            return
        clause = self._active_clause
        if not clause.text.strip():
            self._active_clause = None
            return
        chapter_id = self._chapter_key_from_clause(clause.clause_id)
        chapter = self._ensure_chapter(chapter_id, page=clause.page)
        chapter.clauses.append(
            Clause(clause_id=clause.clause_id, text=clause.text, page=clause.page)
        )
        self._active_chapter_id = chapter_id
        self._active_clause = None

    def _start_clause(self, clause_id: str, page: int, title: str = "") -> None:
        self._flush_clause()
        self._active_clause = _ActiveClause(clause_id=clause_id, page=page, title_part=title)

    def _append_body(self, page: int, text: str) -> None:
        if self._active_clause:
            self._active_clause.body_lines.append(text)
            return
        if self._active_chapter_id:
            chapter = self.chapters[self._active_chapter_id]
            if chapter.clauses:
                last = chapter.clauses[-1]
                last.text = f"{last.text} {text}".strip()
                return

    def process_line(self, page: int, line: str) -> None:
        p = self.patterns

        match = p.gb_chapter.match(line)
        if match:
            self._flush_clause()
            num = cn_numeral_to_int(match.group(1))
            title = match.group(2).strip()
            chapter_id = f"第{num}章" if self.doc_type == DOC_TYPE_CN_GB else num
            if self.doc_type == DOC_TYPE_CN_GB:
                self._chapter_aliases[num] = chapter_id
            self._ensure_chapter(chapter_id, title, page=page)
            self._active_chapter_id = chapter_id
            return

        match = p.iso_chapter.match(line)
        if match:
            self._flush_clause()
            chapter_id = match.group(1)
            title = match.group(2).strip()
            self._ensure_chapter(chapter_id, title, page=page)
            self._active_chapter_id = chapter_id
            return

        match = p.annex.match(line)
        if match:
            self._flush_clause()
            if self.doc_type == DOC_TYPE_CN_GB:
                letter = match.group(1).upper()
                title = (match.group(2) or "").strip()
            elif self.doc_type in (DOC_TYPE_ISO, DOC_TYPE_IEC):
                letter = match.group(1).upper()
                title = (match.group(2) or "").strip()
            else:
                letter = (match.group(1) or match.group(2)).upper()
                title = (match.group(3) or "").strip()
            chapter_id = f"附录{letter}" if self.doc_type == DOC_TYPE_CN_GB else f"Annex {letter}"
            if self.doc_type == DOC_TYPE_CN_GB:
                self._chapter_aliases[letter] = chapter_id
            self._ensure_chapter(chapter_id, title, page=page)
            self._active_chapter_id = chapter_id
            return

        match = p.table.match(line)
        if match:
            if self.doc_type == DOC_TYPE_CN_GB:
                table_id = match.group(1)
                title = (match.group(2) or "").strip()
                clause_id = f"表{table_id}"
            elif self.doc_type in (DOC_TYPE_ISO, DOC_TYPE_IEC):
                table_id = match.group(1)
                title = (match.group(2) or "").strip()
                clause_id = f"Table {table_id}"
            else:
                table_id = match.group(1) or match.group(2)
                title = (match.group(3) or "").strip()
                clause_id = f"Table {table_id}"
            if title and _TABLE_REFERENCE.match(title):
                if self._append_to_existing_clause(clause_id, line):
                    return
            self._start_clause(clause_id, page, title or line)
            return

        match = p.annex_clause.match(line)
        if match:
            clause_id, title = match.group(1), match.group(2).strip()
            norm_id = clause_id.upper() if clause_id[0].isalpha() else clause_id
            self._start_clause(norm_id, page, title)
            chapter_id = self._chapter_key_from_clause(clause_id)
            self._ensure_chapter(chapter_id, page=page)
            self._active_chapter_id = chapter_id
            return

        match = p.clause.match(line)
        if match:
            clause_id, title = match.group(1), match.group(2).strip()
            self._start_clause(clause_id, page, title)
            chapter_id = self._chapter_key_from_clause(clause_id)
            self._ensure_chapter(chapter_id, page=page)
            self._active_chapter_id = chapter_id
            return

        match = p.top_level.match(line)
        if match and is_valid_top_level(
            line, in_body=True, active_chapter_id=self._active_chapter_id
        ):
            self._flush_clause()
            chapter_id, title = match.group(1), match.group(2).strip()
            resolved = self._resolve_chapter_id(chapter_id)
            if resolved in self.chapters or f"第{chapter_id}章" not in self.chapters:
                chapter_id = resolved
            self._ensure_chapter(chapter_id, title, page=page)
            self._active_chapter_id = chapter_id
            self._start_clause(chapter_id, page, title)
            return

        self._append_body(page, line)

    def _append_to_existing_clause(self, clause_id: str, text: str) -> bool:
        if self._active_clause and self._active_clause.clause_id == clause_id:
            self._active_clause.body_lines.append(text)
            return True
        for cid in reversed(self.chapter_order):
            chapter = self.chapters.get(cid)
            if not chapter:
                continue
            for clause in reversed(chapter.clauses):
                if clause.clause_id == clause_id:
                    clause.text = f"{clause.text} {text}".strip()
                    return True
        return False

    def finalize(self) -> list[Chapter]:
        self._flush_clause()
        return [self.chapters[cid] for cid in self.chapter_order if cid in self.chapters]
