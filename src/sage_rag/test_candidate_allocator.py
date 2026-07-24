"""Smoke test for CandidateAllocator / SAGE v3.

Usage:
  python src/sage_rag/test_candidate_allocator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval.bm25 import BM25Retriever
from src.sage_rag.expansion.graph_expander import GraphExpander
from src.sage_rag.graph.graph_store import GraphStore
from src.sage_rag.retrieval.candidate_allocator import CandidateAllocator
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever
from src.sage_rag.retrieval.sage_retriever_v3 import SageRetrieverV3

QUERY = "除零错误审计要求是什么"
GRAPH_DIR = _REPO_ROOT / "data" / "sage_graph"
BM25_DIR = _REPO_ROOT / "data" / "bm25_index"


def _preview(text: str, n: int = 64) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def _show(title: str, units, limit: int = 12) -> None:
    print(f"\n{title} (n={len(units)})")
    for i, u in enumerate(units[:limit], start=1):
        meta = u.metadata or {}
        print(
            f"  {i}. [{meta.get('candidate_source')}/{meta.get('expansion_relation')}] "
            f"alloc={meta.get('allocation_source')} budget={meta.get('graph_budget')} "
            f"{u.unit_id}  {_preview(u.text)}"
        )


def main() -> int:
    store = GraphStore.from_dir(GRAPH_DIR)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=BM25_DIR)
    expansion = SageExpansionRetriever(bm25, expander)
    allocator = CandidateAllocator(fixed_graph_budget=3)

    print("=" * 72)
    print(f"Query: {QUERY}")
    print("=" * 72)

    bm25_top = bm25.retrieve(QUERY, top_k=10)
    _show("1. BM25 Top10", bm25_top)

    pool = expansion.retrieve(QUERY, top_k=80, initial_k=10)
    expanded = [u for u in pool if (u.metadata or {}).get("candidate_source") == "expanded"]
    _show("2. Expanded candidates", expanded, limit=15)

    for strat in ("none", "fixed", "adaptive"):
        allocated = allocator.allocate(pool, query=QUERY, top_k=10, strategy=strat)
        graph_n = sum(
            1
            for u in allocated
            if (u.metadata or {}).get("allocation_source") == "graph_reserved"
        )
        print(f"\n3. Allocation strategy={strat!r} → size={len(allocated)}, graph_reserved={graph_n}")
        _show(f"   allocated ({strat})", allocated, limit=10)

    sage = SageRetrieverV3(bm25, expander, strategy="adaptive")
    final = sage.retrieve(QUERY, top_k=10, initial_k=10)
    print("\n4. SAGE v3 adaptive final Top10")
    for u in final:
        meta = u.metadata or {}
        print(
            f"  #{u.rank} alloc={meta.get('allocation_source')} "
            f"src={meta.get('candidate_source')} rel={meta.get('expansion_relation')} "
            f"final={meta.get('final_score')}  {u.unit_id}"
        )

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
