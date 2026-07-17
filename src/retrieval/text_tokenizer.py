"""Mixed Chinese-English tokenizer for sparse retrieval baselines."""

from __future__ import annotations

import re
from typing import Iterable

try:
    import jieba
except ImportError:  # pragma: no cover - optional at import time
    jieba = None  # type: ignore[assignment]

# Split text into CJK runs vs non-CJK runs.
_SEGMENT_PATTERN = re.compile(
    r"([\u4e00-\u9fff]+|[^\u4e00-\u9fff\s]+)",
    flags=re.UNICODE,
)
_LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", flags=re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def contains_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese-English technical text.

    - Chinese segments: jieba word segmentation (search mode).
    - English / numeric segments: lowercase alphanumeric tokens.
    """
    if not text or not text.strip():
        return []

    tokens: list[str] = []
    for segment in _SEGMENT_PATTERN.findall(text.strip()):
        if not segment:
            continue
        if contains_cjk(segment):
            tokens.extend(_tokenize_cjk(segment))
        else:
            tokens.extend(_tokenize_latin(segment))
    return tokens


def tokenize_batch(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize(text) for text in texts]


def _tokenize_cjk(segment: str) -> list[str]:
    if jieba is None:
        raise ImportError(
            "jieba is required for Chinese tokenization. Install with: pip install jieba"
        )
    return [
        token.strip().lower()
        for token in jieba.cut_for_search(segment)
        if token.strip()
    ]


def _tokenize_latin(segment: str) -> list[str]:
    return [match.group(0).lower() for match in _LATIN_TOKEN_PATTERN.finditer(segment)]
