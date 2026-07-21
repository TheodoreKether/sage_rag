#!/usr/bin/env python
"""Build LightRAG index from Evidence Units (custom chunks).

Usage (conda env: lightrag):
  conda activate lightrag
  cd baselines/lightrag
  python scripts/build_index.py
  python scripts/build_index.py --limit-docs 1 --limit-units 5   # smoke
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BASELINE_ROOT.parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.ingest import ingest_evidence_units  # noqa: E402
from adapter.paths import EVIDENCE_UNITS, RAG_STORAGE_DIR  # noqa: E402
from adapter.rag_factory import create_rag  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest Evidence Units into LightRAG")
    p.add_argument("--evidence", type=Path, default=EVIDENCE_UNITS)
    p.add_argument("--working-dir", type=Path, default=RAG_STORAGE_DIR)
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument("--limit-units", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


async def _main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rag = await create_rag(working_dir=args.working_dir)
    try:
        manifest = await ingest_evidence_units(
            rag,
            evidence_path=args.evidence,
            limit_docs=args.limit_docs,
            limit_units=args.limit_units,
        )
        print(json_dumps(manifest))
    finally:
        await rag.finalize_storages()
    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ns = build_parser().parse_args()
    raise SystemExit(asyncio.run(_main(ns)))
