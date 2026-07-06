"""Construct QA dataset records from Evidence Units via LLM generation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .llm_interface import LLMGenerator, extract_json_array
from .prompt_template import QUESTION_TYPES, render_qa_prompt
from .qa_validator import QAValidator, ValidationReport

logger = logging.getLogger(__name__)

MIN_EVIDENCE_CHARS = 30


@dataclass
class EvidenceUnit:
    unit_id: str
    document_id: str
    document_type: str
    title: str
    chapter_id: str
    chapter_title: str
    parent_clause: str
    page: int
    text: str
    token_length: int
    char_length: int
    split_index: int
    split_total: int
    metadata: dict[str, bool]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceUnit:
        return cls(
            unit_id=data["unit_id"],
            document_id=data["document_id"],
            document_type=data["document_type"],
            title=data.get("title", ""),
            chapter_id=data.get("chapter_id", ""),
            chapter_title=data.get("chapter_title", ""),
            parent_clause=data.get("parent_clause", ""),
            page=int(data.get("page") or 0),
            text=data.get("text", ""),
            token_length=int(data.get("token_length") or 0),
            char_length=int(data.get("char_length") or 0),
            split_index=int(data.get("split_index") or 1),
            split_total=int(data.get("split_total") or 1),
            metadata=data.get("metadata") or {},
        )


@dataclass
class BuildStats:
    evidence_units_read: int = 0
    evidence_units_skipped: int = 0
    raw_pairs_generated: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    llm_errors: int = 0
    by_question_type: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)
    validation: ValidationReport | None = None


def load_evidence_units(path: Path) -> Iterator[EvidenceUnit]:
    jsonl_path = path
    if path.is_dir():
        jsonl_path = path / "evidence_units.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Evidence units file not found: {jsonl_path}")

    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield EvidenceUnit.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed evidence line %d: %s", line_no, exc)


def make_qa_id(unit_id: str, question_type: str, index: int) -> str:
    digest = hashlib.sha1(f"{unit_id}::{question_type}::{index}".encode()).hexdigest()[:10]
    return f"{unit_id}::qa::{question_type}::{index}::{digest}"


def select_question_types(
    configured: list[str],
    *,
    pairs_per_unit: int,
    rng: random.Random,
) -> list[str]:
    if pairs_per_unit <= 0:
        return []
    pool = configured or list(QUESTION_TYPES)
    if pairs_per_unit >= len(pool):
        chosen = list(pool)
        while len(chosen) < pairs_per_unit:
            chosen.append(rng.choice(pool))
        return chosen[:pairs_per_unit]
    return rng.sample(pool, pairs_per_unit)


def build_qa_record(
    *,
    unit: EvidenceUnit,
    question: str,
    answer: str,
    difficulty: str,
    question_type: str,
    qa_index: int,
) -> dict[str, Any]:
    return {
        "qa_id": make_qa_id(unit.unit_id, question_type, qa_index),
        "question": question.strip(),
        "answer": answer.strip(),
        "supporting_evidence": [
            {
                "unit_id": unit.unit_id,
                "document_id": unit.document_id,
                "parent_clause": unit.parent_clause,
            }
        ],
        "difficulty": difficulty,
        "question_type": question_type,
        "document_type": unit.document_type,
        "document_id": unit.document_id,
    }


def generate_pairs_for_unit(
    unit: EvidenceUnit,
    *,
    llm: LLMGenerator,
    question_types: list[str],
    prompt_file: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, qtype in enumerate(question_types, start=1):
        prompt = render_qa_prompt(
            question_type=qtype,
            num_pairs=1,
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            document_type=unit.document_type,
            title=unit.title,
            chapter_id=unit.chapter_id,
            chapter_title=unit.chapter_title,
            parent_clause=unit.parent_clause,
            evidence_text=unit.text,
            prompt_file=prompt_file,
        )
        raw = llm.generate(prompt)
        items = extract_json_array(raw)
        for item in items[:1]:
            records.append(
                build_qa_record(
                    unit=unit,
                    question=str(item.get("question", "")),
                    answer=str(item.get("answer", "")),
                    difficulty=str(item.get("difficulty", "medium")),
                    question_type=qtype,
                    qa_index=idx,
                )
            )
    return records


def build_qa_dataset(
    input_path: Path,
    *,
    llm: LLMGenerator,
    question_types: list[str] | None = None,
    pairs_min: int = 1,
    pairs_max: int = 3,
    prompt_file: str | None = None,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], BuildStats]:
    rng = random.Random(seed)
    configured_types = question_types or list(QUESTION_TYPES)
    stats = BuildStats()
    raw_records: list[dict[str, Any]] = []

    units = list(load_evidence_units(input_path))
    if limit is not None:
        units = units[:limit]

    for unit in tqdm(units, desc="Generating QA pairs", unit="unit"):
        stats.evidence_units_read += 1
        if len(unit.text.strip()) < MIN_EVIDENCE_CHARS:
            stats.evidence_units_skipped += 1
            continue

        n_pairs = rng.randint(pairs_min, pairs_max)
        chosen_types = select_question_types(configured_types, pairs_per_unit=n_pairs, rng=rng)

        try:
            pairs = generate_pairs_for_unit(
                unit,
                llm=llm,
                question_types=chosen_types,
                prompt_file=prompt_file,
            )
            stats.raw_pairs_generated += len(pairs)
            raw_records.extend(pairs)
        except Exception as exc:
            stats.llm_errors += 1
            logger.warning("Generation failed for %s: %s", unit.unit_id, exc)

    validator = QAValidator()
    validation = validator.validate_batch(raw_records)
    stats.validation = validation
    stats.accepted_pairs = validation.accepted_count
    stats.rejected_pairs = validation.rejected_count

    for record in validation.accepted:
        stats.by_question_type[record["question_type"]] = (
            stats.by_question_type.get(record["question_type"], 0) + 1
        )
        stats.by_difficulty[record["difficulty"]] = (
            stats.by_difficulty.get(record["difficulty"], 0) + 1
        )

    return validation.accepted, stats


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d QA pairs to %s", len(records), output_path)


def render_quality_report(
    stats: BuildStats,
    *,
    output_path: Path,
) -> None:
    validation = stats.validation
    lines = [
        "# QA Dataset Quality Report",
        "",
        "## Build Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Evidence units read | {stats.evidence_units_read} |",
        f"| Evidence units skipped (too short) | {stats.evidence_units_skipped} |",
        f"| Raw QA pairs generated | {stats.raw_pairs_generated} |",
        f"| Accepted QA pairs | {stats.accepted_pairs} |",
        f"| Rejected QA pairs | {stats.rejected_pairs} |",
        f"| LLM errors | {stats.llm_errors} |",
        "",
        "## Question Type Distribution",
        "",
        "| question_type | count |",
        "|---------------|------:|",
    ]
    for qtype, count in sorted(stats.by_question_type.items()):
        lines.append(f"| {qtype} | {count} |")

    lines.extend(["", "## Difficulty Distribution", "", "| difficulty | count |", "|------------|------:|"])
    for diff, count in sorted(stats.by_difficulty.items()):
        lines.append(f"| {diff} | {count} |")

    if validation and validation.rejected:
        lines.extend(["", "## Rejected Records", ""])
        for item in validation.rejected[:50]:
            lines.append(
                f"- `{item.get('qa_id', '')}`: {item.get('reason')} — "
                f"{item.get('question_preview', '')}"
            )
        if len(validation.rejected) > 50:
            lines.append(f"- ... and {len(validation.rejected) - 50} more")

    if stats.accepted_pairs:
        # placeholder for avg lengths if we stored them - skip for now
        lines.extend(["", "## Notes", "", "Duplicate questions and empty fields were removed by QAValidator."])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote quality report to %s", output_path)
