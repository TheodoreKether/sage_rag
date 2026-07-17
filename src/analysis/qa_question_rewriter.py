"""Rewrite misaligned QA questions from gold evidence content."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

_CODE_EXAMPLE_SUFFIX = re.compile(r"代码示例\s*$")
_GENERIC_CN_TOPICS = {
    "范围",
    "规范性引用文件",
    "术语和定义",
    "缩略语",
    "概述",
    "总体流程",
    "信息模型概述",
    "数据集成模型",
    "数据分类",
    "(续)",
}

_BAD_TITLE_PAT = re.compile(
    r"20\d{2}.*发布|^GB\s*T\s*\d|copyright|publication|If you have any questions",
    re.IGNORECASE,
)

DOC_DOMAIN_LABELS: dict[str, str] = {
    "GB_T_39401-2020": "工业机器人云服务平台数据交换",
    "GB_T_39412-2020": "代码安全审计",
    "GB_T_39454-2020": "货物跟踪服务",
    "GB_T_39457-2020": "汇付通知过程",
    "IEC_62771-2012": "IEC 62771",
    "IEC_62966-3-2021": "IEC 62966-3",
}

CN_TYPE_TEMPLATES: dict[str, str] = {
    "definition": "如何理解「{topic}」的含义？",
    "requirement": "在合规检查中，「{topic}」有哪些必须遵守的约束？",
    "procedure": "执行「{topic}」相关检查通常包含哪些步骤？",
    "purpose": "为什么要关注「{topic}」？",
    "comparison": "与「{topic}」相关的不同类型或模式之间有何区别？",
    "constraint": "在「{topic}」方面存在哪些限制条件或边界要求？",
    "enumeration": "「{topic}」涉及哪些主要组成要素或数据项？",
    "exception": "在「{topic}」场景下是否存在例外情况或特殊处理？",
    "explanation": "能否概述「{topic}」的工作原理或关键机制？",
    "cause": "「{topic}」可能引发哪些问题，其成因是什么？",
    "application": "在实际项目中，「{topic}」一般如何落地应用？",
}

EN_TYPE_TEMPLATES: dict[str, str] = {
    "definition": "What does `{topic}` mean in this technical context?",
    "requirement": "What requirements apply to `{topic}`?",
    "procedure": "What steps or methods are involved in `{topic}`?",
    "purpose": "What is the purpose of `{topic}`?",
    "comparison": "How do the alternatives related to `{topic}` differ?",
    "constraint": "What limits or boundaries apply to `{topic}`?",
    "enumeration": "What key items or components are associated with `{topic}`?",
    "exception": "Are there exceptions or special cases for `{topic}`?",
    "explanation": "How should `{topic}` be understood in practice?",
    "cause": "What causes issues related to `{topic}`?",
    "application": "Where is `{topic}` typically applied?",
}


def _load_qa_quality():
    path = Path(__file__).resolve().parents[1] / "generation" / "qa_quality.py"
    spec = importlib.util.spec_from_file_location("qa_quality", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load qa_quality from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_qa_quality = _load_qa_quality()
check_natural_question = _qa_quality.check_natural_question


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _first_sentence(text: str) -> str:
    cleaned = _normalize_space(text)
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[。！？；.!?;])\s+", cleaned, maxsplit=1)
    return parts[0].strip()


def _trim_topic(topic: str, *, max_len: int = 24) -> str:
    topic = _CODE_EXAMPLE_SUFFIX.sub("", topic).strip(" ，,;；")
    topic = re.sub(r"\s+", " ", topic)
    if "代码示例" in topic:
        topic = topic.split("代码示例", 1)[0].strip()
    if len(topic) > max_len:
        topic = topic[:max_len].rstrip(" ，,;；")
    return topic


def _is_english_text(text: str) -> bool:
    sample = text[:400]
    cjk = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
    ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
    if cjk >= 8:
        return False
    return ascii_letters >= 24


def extract_cn_topic(evidence_text: str, chapter_title: str = "") -> str:
    text = _normalize_space(evidence_text)
    if not text:
        return chapter_title or "相关内容"

    match = re.match(r"^(.+?)\s*审计指标[:：]", text)
    if match:
        return _trim_topic(match.group(1))

    if "代码示例" in text[:80]:
        return _trim_topic(text.split("代码示例", 1)[0])

    for sep in (
        " 下列",
        " 如图",
        " 其中",
        " 项目 ",
        " 类别 ",
        " 包括",
        " 示例",
        " 主要",
        " 数据项名",
    ):
        if sep in text:
            candidate = _trim_topic(text.split(sep, 1)[0])
            if len(candidate) >= 2:
                return candidate

    first = _first_sentence(text)
    if chapter_title and first.startswith(chapter_title):
        first = first[len(chapter_title) :].strip()

    if "：" in first[:40]:
        left, right = first.split("：", 1)
        if 4 <= len(left) <= 30:
            return _trim_topic(left)
        if 4 <= len(right) <= 30:
            return _trim_topic(right)

    words = first.split()
    if len(words) >= 2:
        head = " ".join(words[:4])
        if 4 <= len(head) <= 36:
            return _trim_topic(head)

    return _trim_topic(first or chapter_title or "相关内容")


def extract_en_topic(evidence_text: str) -> str:
    text = _normalize_space(evidence_text)
    if not text:
        return "this requirement"

    match = re.match(r"^([A-Z][A-Za-z0-9_,\s\-]{4,80}?)\s+The\s+", text)
    if match:
        return _trim_topic(match.group(1), max_len=48)

    match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\b", text)
    if match:
        return match.group(1)

    return _trim_topic(_first_sentence(text), max_len=48)


def extract_evidence_snippet(evidence_text: str, *, max_len: int = 28) -> str:
    text = _normalize_space(evidence_text)
    for marker in ("数据项含义:", "数据项含义：", "审计指标:", "审计指标："):
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            if len(tail) >= 8:
                return tail[:max_len]
    if len(text) > max_len + 10:
        start = max(0, len(text) // 4)
        snippet = text[start : start + max_len].strip(" ，,;；.")
        if len(snippet) >= 8:
            return snippet
    return text[:max_len]


def extract_topic(unit: dict[str, Any], evidence_text: str) -> str:
    chapter = (unit.get("chapter_title") or "").strip()
    if _is_english_text(evidence_text):
        return extract_en_topic(evidence_text)
    topic = extract_cn_topic(evidence_text, chapter)
    if topic in _GENERIC_CN_TOPICS:
        if chapter and chapter not in _GENERIC_CN_TOPICS:
            return _trim_topic(chapter)
        # pull a more specific phrase from evidence body
        body = _normalize_space(evidence_text)
        for marker in ("包括:", "包括：", "是指", "旨在", "用于"):
            if marker in body:
                return _trim_topic(body.split(marker, 1)[0])
    return topic


def _fallback_cn_question(
    question_type: str,
    topic: str,
    domain: str,
    snippet: str,
) -> str:
    if question_type == "explanation":
        return f"从工程实践角度看，{domain}中与「{snippet}」相关的规范说明了什么？"
    if question_type == "application":
        return f"针对{domain}，工程团队应如何理解「{snippet}」相关要求的意图？"
    if question_type == "cause":
        return f"开发或审计{domain}相关系统时，「{snippet}」涉及哪些技术细节？"
    if question_type == "constraint":
        return f"实施{domain}时，涉及「{snippet}」的关键信息有哪些？"
    template = CN_TYPE_TEMPLATES.get(
        question_type, "与「{topic}」相关的技术内容通常需要关注什么？"
    )
    return template.format(topic=topic)


def resolve_domain(unit: dict[str, Any], qa: dict[str, Any]) -> str:
    doc_id = str(qa.get("document_id") or unit.get("document_id") or "")
    if doc_id in DOC_DOMAIN_LABELS:
        return DOC_DOMAIN_LABELS[doc_id]
    title = (unit.get("title") or "").strip()
    if title and not _BAD_TITLE_PAT.search(title) and len(title) <= 40:
        return title
    chapter = (unit.get("chapter_title") or "").strip()
    if chapter:
        return chapter
    return doc_id.replace("_", " ")


def rewrite_question(qa: dict[str, Any], unit: dict[str, Any]) -> str:
    evidence = unit.get("text") or qa.get("answer") or ""
    question_type = str(qa.get("question_type") or "explanation")
    domain = resolve_domain(unit, qa)
    topic = extract_topic(unit, evidence)
    chapter_title = unit.get("chapter_title") or ""
    parent_clause = unit.get("parent_clause") or ""

    if _is_english_text(evidence):
        template = EN_TYPE_TEMPLATES.get(
            question_type,
            "What should engineers know about `{topic}`?",
        )
        question = template.format(topic=topic)
    else:
        template = CN_TYPE_TEMPLATES.get(question_type)
        if template:
            question = template.format(topic=topic)
        else:
            question = _fallback_cn_question(
                question_type,
                topic,
                domain,
                extract_evidence_snippet(evidence),
            )

    ok, _ = check_natural_question(
        question,
        evidence_text=evidence,
        chapter_title=chapter_title,
        parent_clause=parent_clause,
    )
    if ok:
        return question

    if _is_english_text(evidence):
        alt = (
            f"What should engineers know about `{topic}` "
            f"in {qa.get('document_id', 'this standard')}?"
        )
        ok2, _ = check_natural_question(
            alt,
            evidence_text=evidence,
            chapter_title=chapter_title,
            parent_clause=parent_clause,
        )
        return alt if ok2 else question

    alt = _fallback_cn_question(
        question_type,
        topic,
        domain,
        extract_evidence_snippet(evidence),
    )
    ok3, _ = check_natural_question(
        alt,
        evidence_text=evidence,
        chapter_title=chapter_title,
        parent_clause=parent_clause,
    )
    return alt if ok3 else question
