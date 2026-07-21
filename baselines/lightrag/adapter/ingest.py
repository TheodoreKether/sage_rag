"""Ingest Evidence Units into LightRAG as custom chunks (skip official chunker)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .id_map import save_id_maps, save_manifest
from .paths import (
    CHUNK_TO_UNIT_MAP,
    EVIDENCE_UNITS,
    INGEST_MANIFEST,
    UNIT_TO_CHUNK_MAP,
)

logger = logging.getLogger(__name__)


def load_evidence_units(path: Path | None = None) -> list[dict[str, Any]]:
    eu_path = path or EVIDENCE_UNITS
    if not eu_path.is_file():
        raise FileNotFoundError(f"Evidence Units not found: {eu_path}")

    units: list[dict[str, Any]] = []
    with eu_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {eu_path}:{line_no}") from exc
            unit_id = (obj.get("unit_id") or "").strip()
            text = obj.get("text") or ""
            document_id = (obj.get("document_id") or "").strip()
            if not unit_id or not document_id or not str(text).strip():
                logger.warning("Skipping incomplete EU at line %d", line_no)
                continue
            units.append(obj)
    return units


def group_units_by_document(
    units: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[str(unit["document_id"])].append(unit)
    # Stable order within each document (as in jsonl / parent_clause).
    for doc_id in grouped:
        grouped[doc_id].sort(
            key=lambda u: (
                str(u.get("chapter_id") or ""),
                str(u.get("parent_clause") or ""),
                int(u.get("split_index") or 1),
                str(u.get("unit_id") or ""),
            )
        )
    return dict(grouped)


def compute_chunk_id(doc_id: str, text: str) -> str:
    """Mirror LightRAG make_custom_chunk_id after sanitize (import official util)."""
    from lightrag.utils import sanitize_text_for_encoding
    from lightrag.utils_pipeline import make_custom_chunk_id

    cleaned = sanitize_text_for_encoding(text)
    return make_custom_chunk_id(doc_id, cleaned)


async def ingest_evidence_units(
    rag,
    *,
    evidence_path: Path | None = None,
    limit_docs: int | None = None,
    limit_units: int | None = None,
    chunk_map_path: Path = CHUNK_TO_UNIT_MAP,
    unit_map_path: Path = UNIT_TO_CHUNK_MAP,
    manifest_path: Path = INGEST_MANIFEST,
) -> dict[str, Any]:
    """Insert EUs via ainsert_custom_chunks; write id maps.

    Official LightRAG code is untouched; we only call public APIs.
    """
    units = load_evidence_units(evidence_path)
    if limit_units is not None:
        units = units[: max(0, limit_units)]

    grouped = group_units_by_document(units)
    doc_ids = sorted(grouped.keys())
    if limit_docs is not None:
        doc_ids = doc_ids[: max(0, limit_docs)]

    chunk_to_unit: dict[str, str] = {}
    inserted_docs = 0
    inserted_chunks = 0
    collisions = 0

    for doc_id in doc_ids:
        doc_units = grouped[doc_id]
        texts: list[str] = []
        pending_pairs: list[tuple[str, str]] = []  # (chunk_id, unit_id)

        for unit in doc_units:
            text = str(unit["text"])
            unit_id = str(unit["unit_id"])
            chunk_id = compute_chunk_id(doc_id, text)
            if chunk_id in chunk_to_unit and chunk_to_unit[chunk_id] != unit_id:
                collisions += 1
                logger.error(
                    "chunk_id collision: %s -> %s vs %s",
                    chunk_id,
                    chunk_to_unit[chunk_id],
                    unit_id,
                )
                continue
            chunk_to_unit[chunk_id] = unit_id
            texts.append(text)
            pending_pairs.append((chunk_id, unit_id))

        if not texts:
            continue

        full_text = "\n\n".join(texts)
        logger.info(
            "Ingesting doc=%s units=%d ...",
            doc_id,
            len(texts),
        )
        await rag.ainsert_custom_chunks(full_text, texts, doc_id=doc_id)
        inserted_docs += 1
        inserted_chunks += len(texts)
        logger.info("Finished doc=%s", doc_id)

    save_id_maps(
        chunk_to_unit,
        chunk_map_path=chunk_map_path,
        unit_map_path=unit_map_path,
    )
    manifest = {
        "evidence_path": str(evidence_path or EVIDENCE_UNITS),
        "documents": inserted_docs,
        "chunks": inserted_chunks,
        "mapped_chunks": len(chunk_to_unit),
        "collisions": collisions,
        "doc_ids": doc_ids,
        "chunk_map": str(chunk_map_path),
        "unit_map": str(unit_map_path),
    }
    save_manifest(manifest_path, manifest)
    logger.info(
        "Ingest done: docs=%d chunks=%d mapped=%d collisions=%d",
        inserted_docs,
        inserted_chunks,
        len(chunk_to_unit),
        collisions,
    )
    return manifest
