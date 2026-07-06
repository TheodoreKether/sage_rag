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

    def generate(self, prompt: str) -> str:
        fields = self._parse_fields(prompt)
        question_type = self._extract_question_type(prompt)
        num_pairs = int(fields.get("num_pairs", "1") or "1")
        evidence = fields.get("evidence_text", "").strip()
        if not evidence:
            raise ValueError("Placeholder backend: missing evidence_text in prompt")

        pairs = []
        for idx in range(num_pairs):
            pairs.append(
                self._synthesize_pair(
                    question_type=question_type,
                    evidence=evidence,
                    fields=fields,
                    variant=idx,
                )
            )
        return json.dumps(pairs, ensure_ascii=False)

    def _parse_fields(self, prompt: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        lines = prompt.splitlines()
        in_evidence = False
        evidence_lines: list[str] = []

        for line in lines:
            if line.strip() == "## Evidence Unit Text":
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
        match = re.search(r"pair\(s\) of the following type:\s*\*\*(\w+)\*\*", prompt)
        return match.group(1) if match else "definition"

    def _synthesize_pair(
        self,
        *,
        question_type: str,
        evidence: str,
        fields: dict[str, str],
        variant: int,
    ) -> dict[str, str]:
        clause = fields.get("parent_clause", "")
        chapter = fields.get("chapter_title") or fields.get("chapter_id", "")
        doc = fields.get("document_id", "")
        snippet = self._summarize(evidence, max_chars=180)

        templates = {
            "definition": (
                f"根据条款 {clause}，{chapter} 中如何定义或描述相关概念？",
                f"依据 {doc} 条款 {clause}，{snippet}",
            ),
            "requirement": (
                f"标准 {doc} 条款 {clause} 规定了哪些要求或义务？",
                f"该条款要求：{snippet}",
            ),
            "procedure": (
                f"条款 {clause} 中描述了哪些步骤或方法？",
                f"根据条款内容，相关流程或方法包括：{snippet}",
            ),
            "comparison": (
                f"条款 {clause} 中涉及哪些类别或类型之间的区分？",
                f"条款对比说明了：{snippet}",
            ),
            "enumeration": (
                f"请列举条款 {clause} 中提到的主要项目或要素。",
                f"主要包括：{self._extract_list_hint(evidence)}",
            ),
            "constraint": (
                f"条款 {clause} 设定了哪些限制、阈值或约束条件？",
                f"约束条件包括：{snippet}",
            ),
            "exception": (
                f"条款 {clause} 中是否存在例外情况或特殊条件？",
                f"例外或特殊说明：{snippet}",
            ),
        }

        question, answer = templates.get(question_type, templates["definition"])
        if variant > 0:
            question = f"[{question_type}] {question}（变体 {variant + 1}）"

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
