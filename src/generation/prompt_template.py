"""Load and render external prompt templates for QA generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "qa_generation.yaml"

QUESTION_TYPES = (
    "definition",
    "requirement",
    "procedure",
    "comparison",
    "enumeration",
    "constraint",
    "exception",
)


@lru_cache(maxsize=4)
def load_prompt_config(path: str | None = None) -> dict[str, Any]:
    prompt_path = Path(path) if path else DEFAULT_PROMPT_FILE
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    with prompt_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid prompt config format: {prompt_path}")
    return data


def get_question_type_guidance(question_type: str, config: dict[str, Any] | None = None) -> str:
    cfg = config or load_prompt_config()
    guidance_map = cfg.get("question_type_guidance", {})
    if question_type not in guidance_map:
        available = ", ".join(guidance_map.keys())
        raise ValueError(f"Unknown question_type '{question_type}'. Available: {available}")
    return str(guidance_map[question_type]).strip()


def render_qa_prompt(
    *,
    question_type: str,
    num_pairs: int,
    unit_id: str,
    document_id: str,
    document_type: str,
    title: str,
    chapter_id: str,
    chapter_title: str,
    parent_clause: str,
    evidence_text: str,
    prompt_file: str | None = None,
) -> str:
    """Build the full prompt string from external YAML templates."""
    config = load_prompt_config(prompt_file)
    guidance = get_question_type_guidance(question_type, config)

    user_body = config["user_template"].format(
        unit_id=unit_id,
        document_id=document_id,
        document_type=document_type,
        title=title,
        chapter_id=chapter_id,
        chapter_title=chapter_title,
        parent_clause=parent_clause,
        evidence_text=evidence_text.strip(),
        num_pairs=num_pairs,
        question_type=question_type,
        question_type_guidance=guidance,
    )

    system = config.get("system_message", "").strip()
    return f"{system}\n\n---\n\n{user_body}".strip()
