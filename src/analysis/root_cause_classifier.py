"""Paper-quality root-cause classification for retrieval failures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from src.analysis.failure_classifier import (
    CROSS_REF_REGEX,
    UnitInfo,
    doc_family,
    has_cross_reference,
    is_parent_child,
    parse_unit_id,
)

CATEGORY_ORDER = [
    "dataset_annotation_issue",
    "hierarchical_structure_failure",
    "cross_reference_failure",
    "table_information_failure",
    "appendix_failure",
    "version_confusion",
    "multi_clause_reasoning_failure",
    "semantic_similarity_failure",
    "lexical_matching_failure",
]

CATEGORY_LABELS = {
    "dataset_annotation_issue": "Dataset / Annotation Issue",
    "hierarchical_structure_failure": "Hierarchical Structure Failure",
    "cross_reference_failure": "Cross-reference Failure",
    "table_information_failure": "Table Information Failure",
    "appendix_failure": "Appendix Failure",
    "version_confusion": "Version Confusion",
    "multi_clause_reasoning_failure": "Multi-clause Reasoning Failure",
    "semantic_similarity_failure": "Semantic Similarity Failure",
    "lexical_matching_failure": "Lexical Matching Failure",
}

SAGE_SOLUTIONS = {
    "dataset_annotation_issue": "Improve QA generation/validation; exclude from retriever comparison",
    "hierarchical_structure_failure": "Hierarchical Graph over chapter/clause/sub-clause structure",
    "cross_reference_failure": "Cross-reference Graph linking citing clauses to targets",
    "table_information_failure": "Table Nodes with caption and row-level structure",
    "appendix_failure": "Appendix Links from normative body to annex evidence",
    "version_confusion": "Document-family and edition metadata layer",
    "multi_clause_reasoning_failure": "Multi-hop / graph traversal over related clauses",
    "semantic_similarity_failure": "Structure-aware disambiguation beyond embedding similarity",
    "lexical_matching_failure": "Complement sparse lexical matching with semantic / synonym signals",
}

DATE_TOPIC = re.compile(r"20\d{2}[-年/]\d{1,2}|20\d{2}-\d{2}-\d{2}|发布")
GARBAGE_QUOTE = re.compile(
    r"content segment\s*\d+|示除了|ng relationships|s allowing assigning|"
    r"^上述$|^下列$|^\d+\s+\d+\s+\d+\)",
    re.IGNORECASE,
)
DOC_ID_QUOTE = re.compile(
    r"^(?:ISO|IEC|GB(?:/T)?|GB\s*T)[\s_/\-]*[\d\-]+",
    re.IGNORECASE,
)


@dataclass
class RootCauseRecord:
    qa_id: str
    question: str
    question_type: str
    document_id: str
    gold_unit_ids: list[str]
    retrieved_unit_ids: list[str]
    primary_category: str
    root_cause: str
    why_failed: str
    potential_solution: str
    gold_text: str = ""
    hierarchical_subtype: str = ""
    is_dataset_issue: bool = False
    secondary_signals: list[str] = field(default_factory=list)
    dense_hit: bool | None = None
    bm25_hit: bool | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _tokens(text: str) -> set[str]:
    text = (text or "").lower()
    out: set[str] = set()
    for m in re.finditer(r"[a-z0-9_\-]{3,}", text):
        out.add(m.group())
    chars = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(chars) - 1):
        out.add(chars[i : i + 2])
    return out


def _quoted(question: str) -> list[str]:
    return re.findall(r"「([^」]{2,80})」", question) + re.findall(
        r"`([^`]{2,80})`", question
    )


def _looks_like_doc_id(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return bool(DOC_ID_QUOTE.match(t.replace("_", " ")))


def detect_dataset_issue(
    question: str,
    question_type: str,
    gold: UnitInfo,
    doc_title: str = "",
) -> tuple[bool, str]:
    q = (question or "").strip()
    evidence = gold.text or ""
    if not q:
        return True, "Empty question"
    if not evidence.strip():
        return True, "Empty gold evidence"

    quoted = _quoted(q)
    title = (doc_title or getattr(gold, "title", "") or "").strip()
    if title and (
        DATE_TOPIC.search(title)
        or "copyright" in title.lower()
        or "If you have any questions" in title
    ):
        title = ""

    nt, nq, ne = _normalize(title), _normalize(q), _normalize(evidence)
    ne_flex = ne.replace("_", "")

    if DATE_TOPIC.search(q) and any(k in q for k in ("发布", "实施")):
        if not re.search(r"20\d{2}|发布|实施", evidence[:150]):
            return True, "Question uses publish-date metadata as topic; gold is unrelated clause"

    if GARBAGE_QUOTE.search(q):
        return True, "Question contains truncated/garbage topic span not grounded in gold"
    for qt in quoted:
        if GARBAGE_QUOTE.search(qt) or len(qt.strip()) <= 2:
            return (
                True,
                f"Question topic 「{qt[:40]}」 is truncated/garbage and not answerable from gold",
            )

    title_in_q = False
    if title and len(title) >= 4:
        if title in q or (nt and nt in nq):
            title_in_q = True
        for qt in quoted:
            if SequenceMatcher(None, _normalize(qt), nt).ratio() >= 0.75:
                title_in_q = True
    if title_in_q:
        title_in_ev = title in evidence or (len(nt) >= 6 and nt[:8] in ne)
        looks_like_example = any(
            m in evidence[:120]
            for m in ("代码示例", "示例1", "示例:", "示例：", "enum{", "malloc", "strcpy")
        )
        if not title_in_ev and looks_like_example:
            return (
                True,
                "Question asks about the whole standard title, but gold is a specific code/example clause",
            )

    for qt in quoted:
        if _looks_like_doc_id(qt):
            continue
        nqt = _normalize(qt)
        nqt_flex = nqt.replace("_", "")
        if len(qt) < 4:
            continue
        if title and SequenceMatcher(None, nqt, nt).ratio() >= 0.75:
            continue
        if (
            nqt in ne
            or nqt_flex in ne_flex
            or nqt[: min(6, len(nqt))] in ne
            or qt.replace(" ", "_").lower() in evidence.lower()
            or qt.replace("_", " ").lower() in evidence.lower()
        ):
            continue
        qtok, etok = _tokens(qt.replace("_", " ")), _tokens(evidence[:700])
        if qtok and len(qtok & etok) / len(qtok) < 0.15 and len(qt) >= 6:
            return True, f"Quoted topic 「{qt[:40]}」 is not supported by gold evidence"

    if question_type == "purpose" and any(
        m in evidence[:80] for m in ("代码示例", "示例1", "示例:")
    ):
        if not any(
            m in evidence for m in ("目的", "用于", "旨在", "为什么", "purpose", "intended")
        ):
            if ("为什么" in q or "关注" in q) and title_in_q:
                return (
                    True,
                    "Purpose-type question about the standard cannot be answered by a code-example gold unit",
                )

    q_clean = q
    for qt in quoted:
        q_clean = q_clean.replace(f"「{qt}」", " ").replace(f"`{qt}`", " ")
    for noise in (
        "为什么要关注",
        "如何理解",
        "的含义",
        "一般如何落地应用",
        "在实际项目中",
        "相关的技术内容通常需要关注什么",
        "在合规检查中",
        "有哪些必须遵守的约束",
        "What does",
        "mean in this technical context",
        "What should engineers know about",
        "Are there exceptions or special cases for",
        "What limits or boundaries apply to",
        "What causes issues related to",
        "How does",
        "work?",
    ):
        q_clean = q_clean.replace(noise, " ")
    qtok = _tokens(q_clean.replace("_", " "))
    if title:
        qtok -= _tokens(title)
    qtok = {t for t in qtok if not re.match(r"^(iso|iec|gb)\d*$", t)}
    etok = _tokens(evidence.replace("_", " "))
    if len(qtok) >= 5:
        overlap = len(qtok & etok) / len(qtok)
        if overlap < 0.05 and quoted and any(
            GARBAGE_QUOTE.search(qt) or len(qt.strip()) <= 3 for qt in quoted
        ):
            return True, f"Question and gold have very low topical overlap ({overlap:.2f})"

    return False, ""


def _top_same_doc_hits(
    gold: UnitInfo,
    retrieved_ids: list[str],
    unit_index: dict[str, UnitInfo],
) -> list[str]:
    same: list[str] = []
    for rid in retrieved_ids:
        info = unit_index.get(rid)
        doc = info.document_id if info else parse_unit_id(rid)["document_id"]
        if doc == gold.document_id:
            same.append(rid)
    return same


def classify_retrieval_root_cause(
    *,
    gold: UnitInfo,
    retrieved_ids: list[str],
    unit_index: dict[str, UnitInfo],
    gold_unit_ids: list[str],
    dense_hit: bool | None = None,
    bm25_hit: bool | None = None,
    analyzing_retriever: str = "hybrid",
) -> tuple[str, str, str, str, list[str]]:
    secondary: list[str] = []
    if gold.is_table_like:
        secondary.append("table_signal")
    if gold.is_appendix_like:
        secondary.append("appendix_signal")
    if has_cross_reference(gold.text):
        secondary.append("cross_ref_signal")

    if analyzing_retriever == "bm25" and bm25_hit is False and dense_hit is True:
        return (
            "lexical_matching_failure",
            "",
            "Dense retrieves gold via semantic similarity, but BM25 misses due to lexical/synonym gap",
            "Sparse matching fails on paraphrases or synonym expressions present in the question",
            secondary,
        )

    top_id = retrieved_ids[0] if retrieved_ids else ""
    top_doc = parse_unit_id(top_id)["document_id"] if top_id else ""

    for rid in retrieved_ids[:3]:
        rdoc = parse_unit_id(rid)["document_id"]
        if rdoc != gold.document_id and doc_family(rdoc) == doc_family(gold.document_id):
            return (
                "version_confusion",
                "",
                f"Retrieved edition {rdoc} instead of gold edition {gold.document_id}",
                "Flat retrieval lacks edition/family disambiguation for multi-version standards",
                secondary,
            )

    if gold.is_table_like:
        return (
            "table_information_failure",
            "",
            "Gold evidence is tabular (table unit / contains_table); text retrieval missed it",
            "Flat text chunks poorly represent table structure, captions, and cell semantics",
            secondary,
        )
    if gold.is_appendix_like:
        return (
            "appendix_failure",
            "",
            "Gold evidence resides in appendix/annex content",
            "Annex content is structurally distant from normative questions in flat retrieval",
            secondary,
        )

    if has_cross_reference(gold.text) and CROSS_REF_REGEX.search(gold.text):
        return (
            "cross_reference_failure",
            "",
            "Gold text contains explicit cross-references (See Clause / Annex / Table) that retrieval did not follow",
            "Retriever ranks local wording but does not traverse reference edges to related evidence",
            secondary,
        )

    if len(gold_unit_ids) > 1:
        return (
            "multi_clause_reasoning_failure",
            "",
            "Answer requires multiple gold evidence units jointly",
            "Single-hit Top-K retrieval cannot assemble multi-clause evidence",
            secondary,
        )

    same_doc = _top_same_doc_hits(gold, retrieved_ids, unit_index)
    if same_doc:
        top_same = same_doc[0]
        hier = is_parent_child(gold.parent_clause, parse_unit_id(top_same)["parent_clause"])
        if hier:
            ret_clause = parse_unit_id(top_same)["parent_clause"]
            return (
                "hierarchical_structure_failure",
                hier,
                f"Gold clause {gold.parent_clause} vs retrieved clause {ret_clause} ({hier})",
                "Retriever returns structurally adjacent clause but not the exact node in the hierarchy",
                secondary,
            )
        return (
            "semantic_similarity_failure",
            "",
            "Same document: semantically related wrong clause retrieved",
            "Embedding/lexical similarity confuses neighboring requirements in the same domain",
            secondary,
        )

    if analyzing_retriever == "dense" and bm25_hit is True and dense_hit is False:
        return (
            "semantic_similarity_failure",
            "",
            "BM25 finds gold via exact terms, but dense embedding ranks semantically similar wrong clauses",
            "Dense similarity lacks the lexical precision needed for exact clause grounding",
            secondary,
        )

    if top_doc and top_doc != gold.document_id:
        return (
            "semantic_similarity_failure",
            "",
            f"Retrieved document {top_doc} instead of gold {gold.document_id}",
            "Cross-document topical similarity without document-scope constraints",
            secondary,
        )

    return (
        "semantic_similarity_failure",
        "",
        "Semantically related but incorrect evidence ranked above gold",
        "Similarity scoring lacks structural precision for clause-level grounding",
        secondary,
    )


def classify_failure(
    rec: dict[str, Any],
    *,
    unit_index: dict[str, UnitInfo],
    qa_index: dict[str, dict[str, Any]],
    analyzing_retriever: str,
    dense_hit: bool | None = None,
    bm25_hit: bool | None = None,
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
        parsed = parse_unit_id(gold_ids[0])
        gold = UnitInfo(
            unit_id=gold_ids[0],
            document_id=parsed["document_id"],
            parent_clause=parsed["parent_clause"],
            chapter_id=parsed["chapter_id"],
            chapter_title="",
            text=str(qa.get("answer") or ""),
        )

    question = str(rec.get("question") or qa.get("question") or "")
    qtype = str(rec.get("question_type") or qa.get("question_type") or "")

    is_issue, issue_reason = detect_dataset_issue(
        question,
        qtype,
        gold,
        doc_title=gold.title or "",
    )
    if is_issue:
        return RootCauseRecord(
            qa_id=str(rec.get("qa_id", "")),
            question=question,
            question_type=qtype,
            document_id=str(rec.get("document_id") or gold.document_id),
            gold_unit_ids=gold_ids,
            retrieved_unit_ids=retrieved_ids,
            primary_category="dataset_annotation_issue",
            root_cause=issue_reason,
            why_failed="Not attributed to retriever — question/gold annotation mismatch",
            potential_solution=SAGE_SOLUTIONS["dataset_annotation_issue"],
            gold_text=gold.text[:500],
            is_dataset_issue=True,
            dense_hit=dense_hit,
            bm25_hit=bm25_hit,
        )

    cat, subtype, root, why, secondary = classify_retrieval_root_cause(
        gold=gold,
        retrieved_ids=retrieved_ids,
        unit_index=unit_index,
        gold_unit_ids=gold_ids,
        dense_hit=dense_hit,
        bm25_hit=bm25_hit,
        analyzing_retriever=analyzing_retriever,
    )
    return RootCauseRecord(
        qa_id=str(rec.get("qa_id", "")),
        question=question,
        question_type=qtype,
        document_id=str(rec.get("document_id") or gold.document_id),
        gold_unit_ids=gold_ids,
        retrieved_unit_ids=retrieved_ids,
        primary_category=cat,
        root_cause=root,
        why_failed=why,
        potential_solution=SAGE_SOLUTIONS.get(cat, ""),
        gold_text=gold.text[:500],
        hierarchical_subtype=subtype,
        is_dataset_issue=False,
        secondary_signals=secondary,
        dense_hit=dense_hit,
        bm25_hit=bm25_hit,
    )
