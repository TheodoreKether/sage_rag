#!/usr/bin/env python
"""Lightweight smoke: import LightRAG + compute one chunk id + optional tiny ingest.

  python scripts/smoke_test.py              # import / id map only
  python scripts/smoke_test.py --live       # needs .env; 1 doc / few units
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BASELINE_ROOT.parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def check_imports() -> None:
    import lightrag
    from lightrag import LightRAG, QueryParam
    from lightrag.utils_pipeline import make_custom_chunk_id

    from adapter.ingest import compute_chunk_id, load_evidence_units
    from adapter.paths import EVIDENCE_UNITS, OFFICIAL_ROOT

    print(f"lightrag version: {lightrag.__version__}")
    print(f"official root exists: {OFFICIAL_ROOT.is_dir()}")
    print(f"LightRAG / QueryParam / make_custom_chunk_id OK")
    units = load_evidence_units(EVIDENCE_UNITS)
    u0 = units[0]
    cid = compute_chunk_id(u0["document_id"], u0["text"])
    print(f"sample unit_id={u0['unit_id']}")
    print(f"sample chunk_id={cid}")
    print(f"total evidence units readable: {len(units)}")
    assert cid.startswith("chunk-")
    assert make_custom_chunk_id


async def live_smoke() -> None:
    from adapter.ingest import ingest_evidence_units
    from adapter.paths import BASELINE_ROOT
    from adapter.rag_factory import create_rag
    from adapter.retriever import LightRAGRetriever

    smoke_dir = BASELINE_ROOT / "rag_storage_smoke"
    rag = await create_rag(working_dir=smoke_dir)
    try:
        manifest = await ingest_evidence_units(
            rag,
            limit_docs=1,
            limit_units=3,
            chunk_map_path=BASELINE_ROOT / "maps" / "smoke_chunk_to_unit.json",
            unit_map_path=BASELINE_ROOT / "maps" / "smoke_unit_to_chunk.json",
            manifest_path=BASELINE_ROOT / "maps" / "smoke_manifest.json",
        )
        print("ingest manifest:", manifest)
    finally:
        await rag.finalize_storages()

    retriever = LightRAGRetriever(
        working_dir=smoke_dir,
        chunk_map_path=BASELINE_ROOT / "maps" / "smoke_chunk_to_unit.json",
        mode="naive",
        enable_rerank=False,
    )
    try:
        hits = await retriever._aretrieve("工业机器人 数据交换", top_k=3)
        print(f"retrieve hits: {len(hits)}")
        for h in hits:
            print(f"  rank={h.rank} unit_id={h.unit_id}")
    finally:
        await retriever._ensure_rag()
        if retriever._rag is not None:
            await retriever._rag.finalize_storages()
            retriever._rag = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    check_imports()
    if args.live:
        asyncio.run(live_smoke())
    print("smoke_test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
