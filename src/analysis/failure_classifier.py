"""Failure classification utilities for retrieval benchmark analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CROSS_REF_PATTERNS = [
    r"见(?:条款|附录|表|第\s*\d+)",
    r"参见",
    r"Refer to",
    r"See (?:Clause|Annex|Table|Section)",
    r"as defined in",
    r"according to (?:Clause|Section|Annex)",
    r"GB/T\s*\d",
    r"ISO[\s/]*IEC?\s*\d",
    r"IEC\s*\d",
    r"附录\s*[A-Z]",
    r"Annex\s+[A-Z]",
    r"第\s*\d+(?:\.\d+)*\s*条",
    r"表\s*\d+",
    r"Table\s+\d+",
]

CROSS_REF_REGEX = re.compile("|".join(CROSS_REF_PATTERNS), re.IGNORECASE)


@dataclass
class UnitInfo:
    unit_id: str
    document_id: str
    parent_clause: str
    chapter_id: str
    chapter_title: str
    text: str
    contains_table: bool = False
    contains_appendix: bool = False
    title: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> UnitInfo:
        meta = record.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        text = str(record.get("text") or "")
        chapter_title = str(record.get("chapter_title") or "")
        return cls(
            unit_id=str(record["unit_id"]),
            document_id=str(record.get("document_id") or ""),
            parent_clause=str(record.get("parent_clause") or ""),
            chapter_id=str(record.get("chapter_id") or ""),
            chapter_title=chapter_title,
            text=text,
            contains_table=bool(meta.get("contains_table")),
            contains_appendix=bool(meta.get("contains_appendix")),
            title=str(record.get("title") or ""),
        )

    @property
    def is_table_like(self) -> bool:
        if self.contains_table:
            return True
        clause = self.parent_clause or ""
        if clause.startswith("表") or clause.startswith("Table"):
            return True
        mid = self.unit_id.split("::")
        if len(mid) >= 3 and (mid[2].startswith("表") or mid[2].startswith("Table")):
            return True
        return False

    @property
    def is_appendix_like(self) -> bool:
        if self.contains_appendix:
            return True
        if "Annex" in self.unit_id or "附录" in self.unit_id:
            return True
        markers = ("附录", "Annex", "appendix", "资料性附录", "规范性附录")
        blob = f"{self.chapter_title} {self.text[:200]}"
        return any(m in blob for m in markers)


@dataclass
class FailureRecord:
    qa_id: str
    question: str
    gold_unit_ids: list[str]
    retrieved_unit_ids: list[str]
    primary_category: str
    hierarchical_subtype: str = ""
    reason: str = ""
    gold_text: str = ""
    question_type: str = ""
    document_id: str = ""
    secondary_categories: list[str] = field(default_factory=list)


def parse_unit_id(unit_id: str) -> dict[str, str]:
    parts = unit_id.split("::")
    return {
        "document_id": parts[0] if parts else "",
        "chapter_id": parts[1] if len(parts) > 1 else "",
        "parent_clause": parts[-2] if len(parts) >= 2 else "",
        "clause_path": "::".join(parts[1:-1]) if len(parts) > 2 else "",
    }


def doc_family(document_id: str) -> str:
    return re.sub(r"-\d{4}$", "", document_id)


def clause_prefix(clause: str) -> tuple[str, ...]:
    if not clause or clause.startswith("表"):
        return (clause,)
    parts = [p for p in re.split(r"[.\-]", clause) if p]
    return tuple(parts)


def is_parent_child(gold_clause: str, retrieved_clause: str) -> str | None:
    """Return subtype if hierarchical relation exists."""
    if not gold_clause or not retrieved_clause:
        return None
    if gold_clause == retrieved_clause:
        return None

    g_parts = clause_prefix(gold_clause)
    r_parts = clause_prefix(retrieved_clause)

    if len(g_parts) > len(r_parts) and g_parts[: len(r_parts)] == r_parts:
        return "child_gold_parent_retrieved"
    if len(r_parts) > len(g_parts) and r_parts[: len(g_parts)] == g_parts:
        return "parent_gold_child_retrieved"
    if g_parts[:-1] == r_parts[:-1] and g_parts != r_parts:
        return "sibling_clause_confusion"
    return None


def has_cross_reference(text: str) -> bool:
    return bool(CROSS_REF_REGEX.search(text or ""))


def classify_single_failure(
    *,
    gold: UnitInfo,
    retrieved_ids: list[str],
    unit_index: dict[str, UnitInfo],
    question: str,
) -> tuple[str, str, str, list[str]]:
    """Return primary_category, hierarchical_subtype, reason, secondary_categories."""
    secondary: list[str] = []

    if gold.contains_table:
        secondary.append("table_failure")
    if gold.is_table_like:
        secondary.append("table_failure")
    if gold.is_appendix_like:
        secondary.append("appendix_failure")
    if has_cross_reference(gold.text):
        secondary.append("cross_reference_failure")

    if not retrieved_ids:
        if gold.contains_table or gold.is_table_like:
            return "table_failure", "", "Gold evidence contains a table; retriever returned no hits", secondary
        if gold.is_appendix_like:
            return "appendix_failure", "", "Gold evidence is in appendix; retriever returned no hits", secondary
        if has_cross_reference(gold.text):
            return "cross_reference_failure", "", "Gold cites other clauses; retriever returned no hits", secondary
        return "semantic_failure", "", "No results returned", secondary

    top_id = retrieved_ids[0]
    top_doc = parse_unit_id(top_id)["document_id"]

    if gold.contains_table or gold.is_table_like:
        return (
            "table_failure",
            "",
            "Gold evidence contains a table; retriever missed tabular unit",
            secondary,
        )
    if gold.is_appendix_like:
        return (
            "appendix_failure",
            "",
            "Gold evidence is in appendix/annex content",
            secondary,
        )
    if has_cross_reference(gold.text):
        return (
            "cross_reference_failure",
            "",
            "Gold evidence references other clauses/annexes/tables",
            secondary,
        )

    if gold.document_id != top_doc:
        if doc_family(gold.document_id) == doc_family(top_doc):
            return (
                "version_failure",
                "",
                f"Retrieved {top_doc} instead of gold {gold.document_id}",
                secondary,
            )
        return (
            "cross_document_failure",
            "",
            f"Retrieved {top_doc} instead of gold {gold.document_id}",
            secondary,
        )

    hier = is_parent_child(gold.parent_clause, parse_unit_id(top_id)["parent_clause"])
    if hier:
        return (
            "hierarchical_failure",
            hier,
            f"Gold clause {gold.parent_clause}, top retrieved {parse_unit_id(top_id)['parent_clause']}",
            secondary,
        )

    return (
        "semantic_failure",
        "",
        "Same document but semantically related wrong clause retrieved",
        secondary,
    )


def classify_failure_record(
    rec: dict[str, Any],
    unit_index: dict[str, UnitInfo],
    qa_index: dict[str, dict[str, Any]] | None = None,
) -> FailureRecord | None:
    gold_ids = rec.get("gold_unit_ids") or []
    retrieved_ids = rec.get("retrieved_unit_ids") or []
    if float(rec.get("Recall@10", 0.0)) > 0:
        return None
    if not gold_ids:
        return None

    gold = unit_index.get(gold_ids[0])
    if gold is None:
        gold = UnitInfo(
            unit_id=gold_ids[0],
            document_id=parse_unit_id(gold_ids[0])["document_id"],
            parent_clause=parse_unit_id(gold_ids[0])["parent_clause"],
            chapter_id=parse_unit_id(gold_ids[0])["chapter_id"],
            chapter_title="",
            text="",
        )

    qa = (qa_index or {}).get(rec.get("qa_id", ""), {})
    if not gold.text and qa.get("answer"):
        gold.text = str(qa["answer"])

    primary, subtype, reason, secondary = classify_single_failure(
        gold=gold,
        retrieved_ids=retrieved_ids,
        unit_index=unit_index,
        question=str(rec.get("question") or ""),
    )

    return FailureRecord(
        qa_id=str(rec.get("qa_id", "")),
        question=str(rec.get("question") or ""),
        gold_unit_ids=gold_ids,
        retrieved_unit_ids=retrieved_ids,
        primary_category=primary,
        hierarchical_subtype=subtype,
        reason=reason,
        gold_text=gold.text[:500],
        question_type=str(rec.get("question_type") or qa.get("question_type") or ""),
        document_id=str(rec.get("document_id") or gold.document_id),
        secondary_categories=secondary,
    )


def detect_lexical_failures(
    dense_records: dict[str, dict[str, Any]],
    bm25_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Queries where Dense succeeds but BM25 fails (Recall@10)."""
    lexical: list[dict[str, Any]] = []
    for qa_id, dense in dense_records.items():
        bm25 = bm25_records.get(qa_id)
        if not bm25:
            continue
        if float(dense.get("Recall@10", 0)) > 0 and float(bm25.get("Recall@10", 0)) == 0:
            lexical.append(
                {
                    "qa_id": qa_id,
                    "question": dense.get("question", ""),
                    "gold_unit_ids": dense.get("gold_unit_ids", []),
                    "dense_top3": (dense.get("retrieved_unit_ids") or [])[:3],
                    "bm25_top3": (bm25.get("retrieved_unit_ids") or [])[:3],
                }
            )
    return lexical
