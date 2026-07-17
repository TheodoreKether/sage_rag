"""Build a BM25 index from Evidence Units."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.bm25_index import build_bm25_index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build BM25 sparse index from Evidence Units JSONL.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/evidence_units/evidence_units.jsonl",
        help="Evidence units JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/bm25_index",
        help="Output directory for BM25 index artifacts",
    )
    parser.add_argument(
        "--report",
        default="results/benchmark/bm25_index_report.md",
        help="Markdown build report path",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def render_report(result, *, input_path: Path) -> str:
    return "\n".join(
        [
            "# BM25 Index Build Report",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Input | `{input_path}` |",
            f"| Documents indexed | {result.documents} |",
            f"| Average token length | {result.avg_doc_length:.1f} |",
            f"| BM25 k1 | {result.k1} |",
            f"| BM25 b | {result.b} |",
            f"| Build time | {result.elapsed_seconds:.2f} s |",
            "",
            "## Output Files",
            "",
            f"| File | Path |",
            f"|------|------|",
            f"| BM25 model | `{result.output_dir / 'bm25_model.pkl'}` |",
            f"| Metadata | `{result.output_dir / 'metadata.json'}` |",
            f"| Config | `{result.output_dir / 'index_config.json'}` |",
            "",
            "## Tokenizer",
            "",
            "- Chinese segments: `jieba.cut_for_search`",
            "- English / numeric segments: lowercase alphanumeric tokens",
            "- Utility: `src.retrieval.text_tokenizer.tokenize`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    report_path = Path(args.report)

    try:
        result = build_bm25_index(
            input_path,
            output_dir,
            k1=args.k1,
            b=args.b,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logging.error("%s", exc)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result, input_path=input_path) + "\n", encoding="utf-8")
    logging.info(
        "Done: %d documents indexed in %.2fs -> %s",
        result.documents,
        result.elapsed_seconds,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
