"""CLI entry point for QA dataset construction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from .llm_interface import create_llm_backend
    from .prompt_template import QUESTION_TYPES
    from .qa_builder import build_qa_dataset, render_quality_report, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.generation.llm_interface import create_llm_backend
    from src.generation.prompt_template import QUESTION_TYPES
    from src.generation.qa_builder import build_qa_dataset, render_quality_report, write_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build QA benchmark dataset from Evidence Units using an LLM backend.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/evidence_units",
        help="Evidence units JSONL file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/qa_dataset",
        help="Output directory (writes qa_pairs.jsonl)",
    )
    parser.add_argument(
        "--llm-backend",
        default="placeholder",
        help="LLM backend name: placeholder, openai, deepseek, qwen, glm, ollama",
    )
    parser.add_argument(
        "--question-types",
        default=",".join(QUESTION_TYPES),
        help="Comma-separated question types to sample from",
    )
    parser.add_argument(
        "--pairs-min",
        type=int,
        default=1,
        help="Minimum QA pairs per evidence unit",
    )
    parser.add_argument(
        "--pairs-max",
        type=int,
        default=3,
        help="Maximum QA pairs per evidence unit",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional custom prompt YAML path",
    )
    parser.add_argument(
        "--quality-report",
        default="results/qa_quality_report.md",
        help="Markdown quality report output path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question-type sampling",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N evidence units (for testing)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    question_types = [t.strip() for t in args.question_types.split(",") if t.strip()]
    invalid = [t for t in question_types if t not in QUESTION_TYPES]
    if invalid:
        logging.error("Unknown question types: %s", invalid)
        return 1

    if args.pairs_min < 1 or args.pairs_max < args.pairs_min:
        logging.error("Invalid pairs range: min=%s max=%s", args.pairs_min, args.pairs_max)
        return 1

    try:
        llm = create_llm_backend(args.llm_backend)
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_file = output_dir / "qa_pairs.jsonl"

    try:
        records, stats = build_qa_dataset(
            input_path,
            llm=llm,
            question_types=question_types,
            pairs_min=args.pairs_min,
            pairs_max=args.pairs_max,
            prompt_file=args.prompt_file,
            seed=args.seed,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    write_jsonl(records, output_file)
    render_quality_report(stats, output_path=Path(args.quality_report))

    logging.info(
        "Done: %d accepted QA pairs from %d evidence units (backend=%s)",
        stats.accepted_pairs,
        stats.evidence_units_read,
        llm.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
