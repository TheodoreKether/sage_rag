"""chunk_id ↔ unit_id mapping helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_id_maps(
    chunk_to_unit: dict[str, str],
    *,
    chunk_map_path: Path,
    unit_map_path: Path,
) -> dict[str, str]:
    """Persist both directions of the id map. Returns unit_id -> chunk_id."""
    chunk_map_path.parent.mkdir(parents=True, exist_ok=True)
    unit_to_chunk = {unit_id: chunk_id for chunk_id, unit_id in chunk_to_unit.items()}

    chunk_map_path.write_text(
        json.dumps(chunk_to_unit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unit_map_path.write_text(
        json.dumps(unit_to_chunk, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return unit_to_chunk


def load_chunk_to_unit(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"chunk_id map not found: {path}. Run build_index.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid chunk map JSON: {path}")
    return {str(k): str(v) for k, v in data.items()}


def load_unit_to_chunk(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"unit_id map not found: {path}. Run build_index.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid unit map JSON: {path}")
    return {str(k): str(v) for k, v in data.items()}


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
