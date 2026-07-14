"""Abstract LLM interface and pluggable backends for QA generation."""

from __future__ import annotations

import abc
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class LLMGenerator(abc.ABC):
    """Backend-agnostic interface for text generation."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Return model output text for a fully rendered prompt."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Parse JSON array from raw LLM output, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")
    return data


class PlaceholderLLMGenerator(LLMGenerator):
    """Deterministic offline backend for pipeline development and testing.

    Parses the rendered prompt to recover evidence metadata and produces
    structured JSON matching the expected LLM response schema.
    Replace with OpenAI / DeepSeek / Qwen / GLM / Ollama backends in production.
    """

    _FIELD_RE = re.compile(r"^-?\s*([\w_]+):\s*(.+)$", re.MULTILINE)

    _BOILERPLATE_RE = re.compile(
        r"copyright|photocopying|microfilm|obtaining additional rights|"
        r"All rights reserved|IEC\s+Central\s+Office",
        re.IGNORECASE,
    )

    def generate(self, prompt: str) -> str:
        fields = self._parse_fields(prompt)
        question_type = self._extract_question_type(prompt)
        num_pairs = int(fields.get("num_pairs", "1") or "1")
        evidence = fields.get("evidence_text", "").strip()
        if not evidence:
            raise ValueError("Placeholder backend: missing evidence_text in prompt")

        retry_match = re.search(r"Regeneration attempt:\s*(\d+)", prompt)
        retry = int(retry_match.group(1)) if retry_match else 0
        unit_id = fields.get("unit_id", "")

        pairs = []
        for idx in range(num_pairs):
            pairs.append(
                self._synthesize_pair(
                    question_type=question_type,
                    evidence=evidence,
                    fields=fields,
                    variant=idx + retry,
                    unit_id=unit_id,
                )
            )
        return json.dumps(pairs, ensure_ascii=False)

    def _parse_fields(self, prompt: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        lines = prompt.splitlines()
        in_evidence = False
        evidence_lines: list[str] = []

        for line in lines:
            if line.strip() in ("## Evidence Unit Text", "## Evidence Unit"):
                in_evidence = True
                continue
            if in_evidence and line.strip().startswith("## "):
                in_evidence = False
            if in_evidence:
                evidence_lines.append(line)
                continue

            match = self._FIELD_RE.match(line.strip())
            if match:
                fields[match.group(1)] = match.group(2).strip()
            elif line.strip().startswith("- standard title:"):
                fields["title"] = line.split(":", 1)[1].strip()

        if evidence_lines:
            fields["evidence_text"] = "\n".join(evidence_lines).strip()

        type_match = re.search(r"\*\*(\w+)\*\*", prompt)
        if type_match:
            fields["question_type"] = type_match.group(1)

        num_match = re.search(r"Generate exactly (\d+)", prompt)
        if num_match:
            fields["num_pairs"] = num_match.group(1)

        return fields

    def _extract_question_type(self, prompt: str) -> str:
        match = re.search(r"type:\s*\*\*(\w+)\*\*", prompt)
        if not match:
            match = re.search(r"\*\*(\w+)\*\*", prompt)
        return match.group(1) if match else "definition"

    def _topic_hint(self, evidence: str, title: str, fields: dict[str, str] | None = None) -> str:
        concept = self._extract_concept(evidence)
        if concept:
            return concept

        if fields:
            domain = self._domain_hint(fields)
            cleaned = re.sub(r"\s+", " ", evidence).strip()
            if self._BOILERPLATE_RE.search(cleaned[:200]) or cleaned.lower().startswith("purpose "):
                return domain
            if re.search(r"[{}();=<>]", cleaned[:40]):
                return domain
            if "本标准" in cleaned[:80]:
                return domain

        cleaned = re.sub(r"\s+", " ", evidence).strip()
        cleaned = re.sub(r"^(?:表\s*\d+|Table\s+\d+|\d+(?:\.\d+)*)\s*", "", cleaned)
        if title and self._BOILERPLATE_RE.search(title):
            title = ""
        if self._BOILERPLATE_RE.search(cleaned[:200]):
            words = cleaned[:80].strip(" .，,;；")
            return words if len(words) >= 8 else (fields and self._domain_hint(fields) or "the described technical content")
        if title and len(title) <= 40 and title not in cleaned[:40] and "本标准" not in title:
            return title
        words = cleaned[:60].strip(" .，,;；")
        if len(words) >= 4 and not re.search(r"[{}();=<>]", words):
            return words
        return fields and self._domain_hint(fields) or "the described technical content"

    @staticmethod
    def _extract_concept(evidence: str) -> str:
        text = evidence.strip()

        label = re.match(r"^([^\n：:审计指标]{2,24})\s+审计指标", text)
        if label:
            return label.group(1).strip()

        audit = re.search(r"审计指标[:：]\s*([^。\n；;]+)", text)
        if audit:
            value = audit.group(1).strip()
            value = re.sub(r"^(?:应|宜|不应|不得|必须)", "", value).strip()
            if 4 <= len(value) <= 36:
                return value

        scope = re.search(r"范围\s+本标准规定了(.+?)。", text)
        if scope:
            return scope.group(1).strip()[:36]

        first_line = text.split("\n", 1)[0].strip()
        first_line = re.sub(r"^(?:表\s*\d+|Table\s+\d+|\d+(?:\.\d+)*)\s*", "", first_line)

        attr = re.match(
            r"^([a-z][a-z0-9_]*)\s+(?:This attribute|This element)",
            first_line,
            re.IGNORECASE,
        )
        if attr:
            return attr.group(1).replace("_", " ")

        purpose = re.match(r"^Purpose\s+(.+?)(?:\.|\s+[A-Z][a-z]+\s+[a-z])", first_line)
        if purpose:
            phrase = purpose.group(1).strip()
            if len(phrase) > 40:
                if " AP " in phrase or phrase.startswith("An AP"):
                    return "application protocol (AP)"
                phrase = phrase[:40].rsplit(" ", 1)[0]
            return phrase

        if 4 <= len(first_line) <= 48:
            if re.search(r"[{}();=<>]", first_line):
                return ""
            if re.search(r"^(范围|规范性引用|术语|前言|Foreword|Scope|Introduction|本标准)\b", first_line):
                return ""
            if PlaceholderLLMGenerator._BOILERPLATE_RE.search(first_line):
                return ""
            if "本标准规定" in first_line:
                return ""
            return first_line
        return ""

    def _domain_hint(self, fields: dict[str, str]) -> str:
        title = (fields.get("title") or "").strip()
        title = re.sub(r"\s+", " ", title)
        if title and len(title) <= 40 and "本标准" not in title and not self._BOILERPLATE_RE.search(title):
            return title[:36]
        doc = (fields.get("document_id") or "technical standard").replace("_", " ")
        return doc[:36]

    def _synthesize_pair(
        self,
        *,
        question_type: str,
        evidence: str,
        fields: dict[str, str],
        variant: int,
        unit_id: str = "",
    ) -> dict[str, str]:
        topic = self._topic_hint(
            evidence,
            fields.get("title") or fields.get("document_id", ""),
            fields,
        )
        domain = self._domain_hint(fields)
        focus = topic if topic and len(topic) >= 3 else domain
        snippet = self._summarize(evidence, max_chars=200)
        seed = hash(f"{unit_id}::{question_type}::{variant}") & 0xFFFF

        cn_templates: dict[str, list[tuple[str, str]]] = {
            "definition": [
                (f"如何理解「{focus}」的含义？", f"依据原文，{snippet}"),
                (f"「{focus}」在相关规范中指的是什么？", f"核心含义如下：{snippet}"),
            ],
            "requirement": [
                (f"针对「{focus}」，实施时应满足哪些关键要求？", f"相关要求包括：{snippet}"),
                (f"在合规检查中，「{focus}」有哪些必须遵守的约束？", f"必须满足：{snippet}"),
            ],
            "procedure": [
                (f"执行「{focus}」相关检查通常包含哪些步骤？", f"主要流程或方法包括：{snippet}"),
                (f"如何开展与「{focus}」相关的审计或验证工作？", f"建议步骤：{snippet}"),
            ],
            "purpose": [
                (f"「{focus}」的设计目标或主要用途是什么？", f"从原文看，{snippet}"),
                (f"为什么要关注「{focus}」？", f"主要目的在于：{snippet}"),
            ],
            "comparison": [
                (f"与「{focus}」相关的不同类型或模式之间有何区别？", f"相关区分包括：{self._extract_list_hint(evidence) or snippet}"),
                (f"在「{focus}」场景下，各类别或方案如何区分？", f"区分要点：{self._extract_list_hint(evidence) or snippet}"),
            ],
            "enumeration": [
                (f"「{focus}」涉及哪些主要组成要素或数据项？", f"主要包括：{self._extract_list_hint(evidence) or snippet}"),
                (f"与「{focus}」相关的关键字段或条目有哪些？", f"关键项包括：{self._extract_list_hint(evidence) or snippet}"),
            ],
            "constraint": [
                (f"在「{focus}」方面存在哪些限制条件或边界要求？", f"约束条件包括：{snippet}"),
                (f"实施「{focus}」时需要遵守哪些边界或阈值？", f"边界要求：{snippet}"),
            ],
            "exception": [
                (f"在「{focus}」场景下是否存在例外情况或特殊处理？", f"例外或特殊说明：{snippet}"),
                (f"哪些情况下「{focus}」的规则可以不适用？", f"例外情形：{snippet}"),
            ],
            "explanation": [
                (f"能否概述「{focus}」的工作原理或关键机制？", f"概述如下：{snippet}"),
                (f"「{focus}」的核心机制应如何理解？", f"机制说明：{snippet}"),
            ],
            "cause": [
                (f"导致「{focus}」相关风险或现象的主要原因有哪些？", f"原因说明：{snippet}"),
                (f"「{focus}」可能引发哪些问题，其成因是什么？", f"成因包括：{snippet}"),
            ],
            "application": [
                (f"「{focus}」通常适用于哪些系统或业务场景？", f"适用场景包括：{snippet}"),
                (f"在实际项目中，「{focus}」一般如何落地应用？", f"应用方式：{snippet}"),
            ],
        }

        en_templates: dict[str, list[tuple[str, str]]] = {
            "definition": [
                (f"What does `{focus}` mean in this technical context?", f"According to the source: {snippet}"),
            ],
            "requirement": [
                (f"What requirements apply to `{focus}`?", f"Key requirements: {snippet}"),
            ],
            "procedure": [
                (f"What steps are involved when working with `{focus}`?", f"Main procedure: {snippet}"),
            ],
            "purpose": [
                (f"What is the purpose of `{focus}`?", f"Purpose: {snippet}"),
            ],
            "comparison": [
                (f"How do the categories related to `{focus}` differ?", f"Differences: {self._extract_list_hint(evidence) or snippet}"),
            ],
            "enumeration": [
                (f"What elements or fields are associated with `{focus}`?", f"Includes: {self._extract_list_hint(evidence) or snippet}"),
            ],
            "constraint": [
                (f"What constraints or limits apply to `{focus}`?", f"Constraints: {snippet}"),
            ],
            "exception": [
                (f"Are there exceptions or special cases for `{focus}`?", f"Exceptions: {snippet}"),
            ],
            "explanation": [
                (f"How does `{focus}` work?", f"Overview: {snippet}"),
            ],
            "cause": [
                (f"What causes issues related to `{focus}`?", f"Causes: {snippet}"),
            ],
            "application": [
                (f"Where is `{focus}` typically applied?", f"Applications: {snippet}"),
            ],
        }

        use_cn = bool(re.search(r"[\u4e00-\u9fff]", focus + evidence[:120]))
        pool = cn_templates.get(question_type) or cn_templates["explanation"]
        if not use_cn:
            pool = en_templates.get(question_type) or en_templates["explanation"]
        question, answer = pool[seed % len(pool)]

        difficulty = "easy" if len(evidence) < 120 else ("hard" if len(evidence) > 600 else "medium")
        return {
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
        }

    @staticmethod
    def _summarize(text: str, max_chars: int = 180) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _extract_list_hint(text: str) -> str:
        items = re.findall(r"[a-z]\)\s*([^;；\n]+)", text)
        if items:
            return "；".join(item.strip() for item in items[:5])
        return PlaceholderLLMGenerator._summarize(text, max_chars=160)

    def synthesize_fallback_pair(
        self,
        *,
        question_type: str,
        evidence: str,
        fields: dict[str, str],
        unit_id: str,
    ) -> dict[str, str]:
        """Last-resort natural question when quality retries are exhausted."""
        domain = self._domain_hint(fields)
        snippet = self._summarize(evidence, max_chars=200)
        seed = hash(f"fallback::{unit_id}::{question_type}") & 0xFFFF
        use_cn = bool(re.search(r"[\u4e00-\u9fff]", evidence[:200]))
        hint = self._unique_content_hint(evidence, unit_id)

        cn_pool = [
            (f"在{domain}相关工程中，与「{hint}」相关的技术内容通常需要关注什么？", f"要点如下：{snippet}"),
            (f"实施{domain}时，涉及「{hint}」的关键信息有哪些？", f"关键信息：{snippet}"),
            (f"从工程实践角度看，{domain}中与「{hint}」相关的规范说明了什么？", f"规范说明：{snippet}"),
            (f"技术人员查阅{domain}时，关于「{hint}」应掌握哪些知识？", f"相关知识：{snippet}"),
            (f"在{domain}应用场景下，「{hint}」相关的技术要求实质是什么？", f"实质内容：{snippet}"),
            (f"针对{domain}，工程团队应如何理解「{hint}」相关要求的意图？", f"意图说明：{snippet}"),
            (f"在{domain}标准体系中，「{hint}」相关的实践要求有哪些？", f"实践要求：{snippet}"),
            (f"开发或审计{domain}相关系统时，「{hint}」涉及哪些技术细节？", f"技术细节：{snippet}"),
        ]
        en_pool = [
            (f"What should engineers know about `{hint}` in {domain}?", f"Key points: {snippet}"),
            (f"What technical details of `{hint}` apply within {domain}?", f"Details: {snippet}"),
            (f"How should practitioners interpret `{hint}` in {domain}?", f"Interpretation: {snippet}"),
            (f"What are the practical implications of `{hint}` in {domain}?", f"Implications: {snippet}"),
            (f"What does {domain} specify about `{hint}`?", f"Specification: {snippet}"),
            (f"What engineering considerations apply to `{hint}` in {domain}?", f"Considerations: {snippet}"),
            (f"How is `{hint}` relevant to {domain} implementations?", f"Relevance: {snippet}"),
            (f"What requirements relate to `{hint}` in the {domain} context?", f"Requirements: {snippet}"),
        ]
        pool = cn_pool if use_cn else en_pool
        question, answer = pool[seed % len(pool)]
        difficulty = "easy" if len(evidence) < 120 else ("hard" if len(evidence) > 600 else "medium")
        return {"question": question, "answer": answer, "difficulty": difficulty}

    @staticmethod
    def _unique_content_hint(evidence: str, unit_id: str) -> str:
        text = re.sub(r"\s+", " ", evidence).strip()
        start = max(0, len(text) // 4)
        chunk = text[start : start + 36].strip(" .，,;；")
        chunk = re.sub(r"^(?:表\s*\d+|Table\s+\d+|\d+(?:\.\d+)*)\s*", "", chunk)
        if len(chunk) >= 6 and not re.search(r"[{}();=<>]", chunk[:20]):
            return chunk[:32]
        tail = unit_id.split("::")[-1]
        return f"content segment {tail}"


class OpenAICompatibleGenerator(LLMGenerator):
    """Optional backend stub for OpenAI-compatible APIs (OpenAI, DeepSeek, etc.).

    Requires `openai` package and environment variables:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (optional, for compatible endpoints)
    - OPENAI_MODEL (optional, default gpt-4o-mini)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        import os

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend requires `pip install openai`. "
                "Use --llm-backend placeholder for offline runs."
            ) from exc

        client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        model = self.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        system, _, user = prompt.partition("\n---\n")
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip() or prompt},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


def create_llm_backend(name: str, **kwargs: Any) -> LLMGenerator:
    """Factory for swappable LLM backends without pipeline changes."""
    key = name.strip().lower()
    registry: dict[str, type[LLMGenerator]] = {
        "placeholder": PlaceholderLLMGenerator,
        "openai": OpenAICompatibleGenerator,
        "deepseek": OpenAICompatibleGenerator,
        "qwen": OpenAICompatibleGenerator,
        "glm": OpenAICompatibleGenerator,
        "ollama": OpenAICompatibleGenerator,
    }
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown llm backend '{name}'. Available: {available}")
    return registry[key](**kwargs)
