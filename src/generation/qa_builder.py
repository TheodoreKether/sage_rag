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

from .llm_interface import LLMGenerator, PlaceholderLLMGenerator, extract_json_array
from .prompt_template import QUESTION_TYPES, QUESTION_TYPE_WEIGHTS, render_qa_prompt
from .qa_validator import QAValidator, ValidationReport, normalize_question, validate_candidate_pair

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
    quality_rejected: int = 0
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


def sample_evidence_units(
    units: list[EvidenceUnit],
    n: int,
    *,
    seed: int = 42,
) -> list[EvidenceUnit]:
    """Randomly sample N evidence units (stable order by index)."""
    if n >= len(units):
        return list(units)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(units)), n))
    return [units[i] for i in indices]


def select_question_types(
    configured: list[str],
    *,
    pairs_per_unit: int,
    rng: random.Random,
) -> list[str]:
    if pairs_per_unit <= 0:
        return []
    pool = configured or list(QUESTION_TYPES)
    weights = [QUESTION_TYPE_WEIGHTS.get(t, 1.0) for t in pool]

    if pairs_per_unit >= len(pool):
        chosen = list(pool)
        while len(chosen) < pairs_per_unit:
            chosen.append(rng.choices(pool, weights=weights, k=1)[0])
        return chosen[:pairs_per_unit]

    # weighted sample without replacement (approximate via shuffle + pick)
    indexed = list(enumerate(pool))
    rng.shuffle(indexed)
    indexed.sort(key=lambda x: rng.random() ** (1.0 / weights[x[0]]), reverse=True)
    return [pool[i] for i, _ in indexed[:pairs_per_unit]]


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
    max_retries: int = 3,
    seen_questions: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = seen_questions if seen_questions is not None else set()
    for idx, qtype in enumerate(question_types, start=1):
        accepted = False
        for attempt in range(max_retries):
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
            if attempt > 0:
                prompt += f"\n\n## Regeneration attempt: {attempt + 1}\n"
            raw = llm.generate(prompt)
            items = extract_json_array(raw)
            if not items:
                continue

            item = items[0]
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            ok, reason = validate_candidate_pair(
                question=question,
                answer=answer,
                unit_id=unit.unit_id,
                evidence_text=unit.text,
                chapter_title=unit.chapter_title,
                parent_clause=unit.parent_clause,
            )
            if not ok:
                logger.debug(
                    "Quality reject (attempt %d/%d) unit=%s type=%s: %s — %s",
                    attempt + 1,
                    max_retries,
                    unit.unit_id,
                    qtype,
                    reason,
                    question[:60],
                )
                continue

            q_norm = normalize_question(question)
            if q_norm in seen:
                logger.debug(
                    "Duplicate reject (attempt %d/%d) unit=%s: %s",
                    attempt + 1,
                    max_retries,
                    unit.unit_id,
                    question[:60],
                )
                continue

            records.append(
                build_qa_record(
                    unit=unit,
                    question=question,
                    answer=answer,
                    difficulty=str(item.get("difficulty", "medium")),
                    question_type=qtype,
                    qa_index=idx,
                )
            )
            seen.add(q_norm)
            accepted = True
            break

        if not accepted:
            fallback = _try_fallback_pair(
                unit=unit,
                llm=llm,
                qtype=qtype,
                qa_index=idx,
                seen=seen,
            )
            if fallback:
                records.append(fallback)
                seen.add(normalize_question(fallback["question"]))
            else:
                logger.warning(
                    "No valid QA after %d attempts for unit=%s type=%s",
                    max_retries,
                    unit.unit_id,
                    qtype,
                )
    return records


def _try_fallback_pair(
    *,
    unit: EvidenceUnit,
    llm: LLMGenerator,
    qtype: str,
    qa_index: int,
    seen: set[str],
) -> dict[str, Any] | None:
    """Generate a safe fallback pair when primary generation fails all retries."""
    if not isinstance(llm, PlaceholderLLMGenerator):
        return None

    fields = {
        "title": unit.title,
        "document_id": unit.document_id,
        "unit_id": unit.unit_id,
    }
    for attempt in range(8):
        item = llm.synthesize_fallback_pair(
            question_type=qtype,
            evidence=unit.text,
            fields=fields,
            unit_id=f"{unit.unit_id}::{attempt}",
        )
        question = item["question"].strip()
        answer = item["answer"].strip()
        ok, _ = validate_candidate_pair(
            question=question,
            answer=answer,
            unit_id=unit.unit_id,
            evidence_text=unit.text,
            chapter_title=unit.chapter_title,
            parent_clause=unit.parent_clause,
        )
        if not ok:
            continue
        q_norm = normalize_question(question)
        if q_norm in seen:
            continue
        return build_qa_record(
            unit=unit,
            question=question,
            answer=answer,
            difficulty=str(item.get("difficulty", "medium")),
            question_type=qtype,
            qa_index=qa_index,
        )
    return None


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
    sample: int | None = None,
    max_retries: int = 3,
    strict_natural_language: bool = True,
) -> tuple[list[dict[str, Any]], BuildStats]:
    rng = random.Random(seed)
    configured_types = question_types or list(QUESTION_TYPES)
    stats = BuildStats()
    raw_records: list[dict[str, Any]] = []

    units = list(load_evidence_units(input_path))
    if sample is not None:
        units = sample_evidence_units(units, sample, seed=seed)
    elif limit is not None:
        units = units[:limit]

    validator = QAValidator(strict_natural_language=strict_natural_language)
    seen_questions: set[str] = set()

    for unit in tqdm(units, desc="Generating QA pairs", unit="unit"):
        stats.evidence_units_read += 1
        if len(unit.text.strip()) < MIN_EVIDENCE_CHARS:
            stats.evidence_units_skipped += 1
            continue

        validator.register_evidence_context(
            unit.unit_id,
            evidence_text=unit.text,
            chapter_title=unit.chapter_title,
            parent_clause=unit.parent_clause,
        )

        n_pairs = rng.randint(pairs_min, pairs_max)
        chosen_types = select_question_types(configured_types, pairs_per_unit=n_pairs, rng=rng)

        try:
            pairs = generate_pairs_for_unit(
                unit,
                llm=llm,
                question_types=chosen_types,
                prompt_file=prompt_file,
                max_retries=max_retries,
                seen_questions=seen_questions,
            )
            stats.raw_pairs_generated += len(pairs)
            raw_records.extend(pairs)
            stats.quality_rejected += max(0, n_pairs - len(pairs))
        except Exception as exc:
            stats.llm_errors += 1
            logger.warning("Generation failed for %s: %s", unit.unit_id, exc)

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
        f"| Quality rejected (per-unit retries exhausted) | {stats.quality_rejected} |",
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
        lines.extend(["", "## Notes", "", "Natural-language quality filters reject template-style or structurally-leaking questions. Duplicate questions and empty fields are also removed."])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote quality report to %s", output_path)
