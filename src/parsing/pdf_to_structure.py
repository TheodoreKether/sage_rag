"""Convert raw standards PDFs into clause-level structured JSON.

Orchestrates extraction, page classification, structure building, and post-processing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

try:
    from .extractors import extract_metadata_from_cover, extract_pages, preprocess_lines
    from .page_classifier import extract_toc_entries, find_body_start_page
    from .patterns import build_patterns
    from .postprocess import extract_title, finalize_structure
    from .structure_builder import DocumentStructure, StructureBuilder
except ImportError:  # direct script: python src/parsing/pdf_to_structure.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.parsing.extractors import extract_metadata_from_cover, extract_pages, preprocess_lines
    from src.parsing.page_classifier import extract_toc_entries, find_body_start_page
    from src.parsing.patterns import build_patterns
    from src.parsing.postprocess import extract_title, finalize_structure
    from src.parsing.structure_builder import DocumentStructure, StructureBuilder

logger = logging.getLogger(__name__)

DOC_TYPE_CN_GB = "CN_GB"
DOC_TYPE_ISO = "ISO"
DOC_TYPE_IEC = "IEC"
DOC_TYPE_ENTERPRISE = "ENTERPRISE"


def detect_doc_type(filename: str) -> str:
    """Classify a PDF by filename heuristics."""
    name = Path(filename).stem.upper().replace("-", "_")
    if re.search(r"GB[_/]?T|\bGB\b", name):
        return DOC_TYPE_CN_GB
    if "IEC" in name:
        return DOC_TYPE_IEC
    if "ISO" in name:
        return DOC_TYPE_ISO
    return DOC_TYPE_ENTERPRISE


def standard_id_from_filename(filename: str) -> str:
    """Derive a human-readable standard identifier from the filename."""
    stem = Path(filename).stem
    cleaned = re.sub(r"_\d{3,6}$", "", stem)
    cleaned = re.sub(r"_(?:en|zh|cn)$", "", cleaned, flags=re.IGNORECASE)
    normalized = cleaned.replace("__", "_")

    patterns = [
        (r"GB[_\-/ ]?T[_\- ]?(\d+(?:\.\d+)?)[_\- ](\d{4})", "GB/T {0}-{1}"),
        (r"GB[_\- ]?(\d+(?:\.\d+)?)[_\- ](\d{4})", "GB {0}-{1}"),
        (r"IEC[_\- ]?(?:TS[_\- ]?)?(\d+(?:[_\-]\d+)*)", "IEC {0}"),
        (r"ISO[_\-/ ]?(\d+(?:[_\-]\d+)*)", "ISO {0}"),
    ]
    upper = normalized.upper().replace(" ", "_")
    for pattern, fmt in patterns:
        match = re.search(pattern, upper)
        if match:
            groups = [g.replace("_", "-") for g in match.groups()]
            return fmt.format(*groups)

    return cleaned.replace("_", " ")


def build_structure(
    pages: list[tuple[int, str]],
    *,
    standard_id: str,
    doc_type: str,
    title: str | None = None,
) -> DocumentStructure:
    patterns = build_patterns(doc_type)
    body_start = find_body_start_page(pages, doc_type)
    toc = extract_toc_entries(pages, body_start)
    metadata = extract_metadata_from_cover(pages, doc_type)

    lines = preprocess_lines(
        pages,
        patterns,
        body_start_page=body_start,
        skip_toc_lines=True,
    )

    builder = StructureBuilder(doc_type)
    for page, line in lines:
        builder.process_line(page, line)
    chapters = builder.finalize()

    structure = DocumentStructure(
        standard_id=standard_id,
        doc_type=doc_type,
        title=title or extract_title(pages, standard_id, doc_type),
        metadata=metadata,
        toc=toc,
        chapters=chapters,
        quality={"body_start_page": body_start},
    )
    return finalize_structure(structure)


def parse_pdf(pdf_path: Path) -> dict:
    """Parse a single PDF into the structured JSON schema."""
    pdf_path = Path(pdf_path)
    doc_type = detect_doc_type(pdf_path.name)
    standard_id = standard_id_from_filename(pdf_path.name)
    logger.info("Parsing %s as %s (%s)", pdf_path.name, doc_type, standard_id)

    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError("PDF contains no pages")

    structure = build_structure(
        pages,
        standard_id=standard_id,
        doc_type=doc_type,
    )
    result = structure.to_dict()
    logger.info(
        "Parsed %s: body@p%d, %d toc entries, %d chapters, %d clauses",
        pdf_path.name,
        result["quality"].get("body_start_page", 0),
        len(result.get("toc", [])),
        len(result["chapters"]),
        result["quality"].get("clause_count", 0),
    )
    if result["quality"].get("warnings"):
        logger.warning("Quality warnings for %s: %s", pdf_path.name, result["quality"]["warnings"])
    return result


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def collect_pdf_paths(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF files found in %s", input_path)
    return pdfs


def process_batch(input_path: Path, output_dir: Path) -> tuple[int, int]:
    """Parse all PDFs under input_path; return (success_count, skip_count)."""
    pdfs = collect_pdf_paths(input_path)
    success = 0
    skipped = 0

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs", unit="file"):
        out_file = output_dir / f"{pdf_path.stem}.json"
        try:
            data = parse_pdf(pdf_path)
            save_json(data, out_file)
            success += 1
            logger.info("Wrote %s", out_file)
        except Exception as exc:
            skipped += 1
            logger.warning(
                "Skipped %s: %s",
                pdf_path.name,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

    return success, skipped


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert standards PDFs into clause-level structured JSON.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a PDF file or directory containing PDFs (e.g. data/raw_pdf)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for JSON files (e.g. data/parsed_json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    success, skipped = process_batch(input_path, output_dir)
    logger.info("Done: %d succeeded, %d skipped", success, skipped)
    return 0 if skipped == 0 or success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
