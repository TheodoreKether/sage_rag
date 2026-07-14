"""Regenerate natural-language QA pairs for retrieval benchmark validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.llm_interface import create_llm_backend
from src.generation.prompt_template import LEGACY_PROMPT_FILE, QUESTION_TYPES
from src.generation.qa_builder import build_qa_dataset, render_quality_report, write_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate natural-language QA pairs from Evidence Units. "
            "Use --sample for a small validation set before full regeneration."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/evidence_units/evidence_units.jsonl",
        help="Evidence units JSONL file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/qa_dataset/qa_pairs_natural_sample50.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Randomly sample N evidence units (default: 50 for validation)",
    )
    parser.add_argument(
        "--llm-backend",
        default="placeholder",
        help="LLM backend: placeholder, openai, deepseek, qwen, glm, ollama",
    )
    parser.add_argument(
        "--pairs-per-unit",
        type=int,
        default=1,
        help="QA pairs per evidence unit (default 1 for validation set)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Regenerate attempts when quality check fails",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Prompt YAML (default: natural-language template)",
    )
    parser.add_argument(
        "--legacy-prompt",
        action="store_true",
        help="Use legacy template-style prompt (not recommended)",
    )
    parser.add_argument(
        "--quality-report",
        default="results/benchmark/qa_quality_report_natural_sample50.md",
        help="Markdown quality report path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for unit sampling and question-type selection",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all evidence units (ignores --sample)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.pairs_per_unit < 1:
        logging.error("--pairs-per-unit must be >= 1")
        return 1

    prompt_file = args.prompt_file
    if args.legacy_prompt:
        prompt_file = str(LEGACY_PROMPT_FILE)

    try:
        llm = create_llm_backend(args.llm_backend)
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    output_path = Path(args.output)
    sample = None if args.full else args.sample

    try:
        records, stats = build_qa_dataset(
            Path(args.input),
            llm=llm,
            question_types=list(QUESTION_TYPES),
            pairs_min=args.pairs_per_unit,
            pairs_max=args.pairs_per_unit,
            prompt_file=prompt_file,
            seed=args.seed,
            sample=sample,
            max_retries=args.max_retries,
            strict_natural_language=not args.legacy_prompt,
        )
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    write_jsonl(records, output_path)
    render_quality_report(stats, output_path=Path(args.quality_report))

    logging.info(
        "Done: %d natural QA pairs (units=%d, backend=%s, output=%s)",
        stats.accepted_pairs,
        stats.evidence_units_read,
        llm.name,
        output_path,
    )
    if records:
        logging.info("Sample question: %s", records[0]["question"][:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
