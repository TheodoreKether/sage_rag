"""Tests for SageExpansionRetriever (BM25 + optional Dense).

Usage:
  python src/sage_rag/test_sage_expansion.py
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
from src.sage_rag.retrieval.sage_expansion_retriever import SageExpansionRetriever

QUERY = "除零错误审计要求是什么"
GRAPH_DIR = _REPO_ROOT / "data" / "sage_graph"
BM25_DIR = _REPO_ROOT / "data" / "bm25_index"
DENSE_DIR = _REPO_ROOT / "data" / "vector_store"


def _preview(text: str, n: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def _print_bm25_run(retriever: SageExpansionRetriever, base: BM25Retriever) -> None:
    print("=" * 72)
    print("Test A: BM25 + SageExpansionRetriever")
    print(f"Query: {QUERY}")
    print("=" * 72)

    initial_k, top_k = 5, 15
    initial = base.retrieve(QUERY, top_k=initial_k)
    print("\n1. BM25 initial results:")
    for u in initial:
        print(f"  rank={u.rank}  score={u.score:.4f}  unit_id={u.unit_id}")
        print(f"         text={_preview(u.text)}")

    # Collect expansion-only view for reporting
    print("\n2. Expanded candidates (per seed, before merge):")
    expanded_rows: list[tuple[str, str, str]] = []
    seen_exp: set[str] = set()
    for seed in initial:
        hits = retriever.graph_expander.expand(seed.unit_id, depth=1)
        for h in hits:
            if h.unit_id in {s.unit_id for s in initial}:
                continue
            if h.unit_id in seen_exp:
                continue
            seen_exp.add(h.unit_id)
            expanded_rows.append((h.unit_id, h.relation, seed.unit_id))
            print(
                f"  unit_id={h.unit_id}  relation={h.relation}  from={seed.unit_id}"
            )
    if not expanded_rows:
        print("  (none)")

    final = retriever.retrieve(QUERY, top_k=top_k, initial_k=initial_k)
    print(f"\n3. Final candidates (top_k={top_k}, initial_k={initial_k}):")
    for u in final:
        src = (u.metadata or {}).get("candidate_source", "?")
        rel = (u.metadata or {}).get("expansion_relation")
        score = u.score if u.score is not None else 0.0
        extra = f"  relation={rel}" if src == "expanded" else ""
        print(
            f"  rank={u.rank}  source={src}  score={score:.4f}  "
            f"unit_id={u.unit_id}{extra}"
        )

    n_init = sum(
        1 for u in final if (u.metadata or {}).get("candidate_source") == "initial"
    )
    n_exp = sum(
        1 for u in final if (u.metadata or {}).get("candidate_source") == "expanded"
    )
    print(f"\n  summary: initial={n_init}, expanded={n_exp}, total={len(final)}")
    assert n_init > 0, "expected at least one initial candidate"
    # Expansion may be empty for some queries; do not hard-fail.
    print("  PASS (BM25 path)")


def _print_dense_run(store: GraphStore) -> bool:
    print("\n" + "=" * 72)
    print("Test B: Dense + SageExpansionRetriever (backend-agnostic check)")
    print("=" * 72)
    try:
        from src.retrieval.dense_retriever import DenseRetriever
    except Exception as exc:
        print(f"  SKIP Dense import: {exc}")
        return False

    if not (DENSE_DIR / "faiss.index").is_file():
        print(f"  SKIP: dense index missing under {DENSE_DIR}")
        return False

    try:
        dense = DenseRetriever(index_dir=DENSE_DIR)
    except Exception as exc:
        print(f"  SKIP Dense init: {exc}")
        return False

    expander = GraphExpander(store)
    sage = SageExpansionRetriever(dense, expander)
    final = sage.retrieve(QUERY, top_k=12, initial_k=4)
    print(f"Query: {QUERY}")
    for u in final:
        src = (u.metadata or {}).get("candidate_source", "?")
        print(
            f"  rank={u.rank}  source={src}  score={u.score}  unit_id={u.unit_id}"
        )
    assert any(
        (u.metadata or {}).get("candidate_source") == "initial" for u in final
    )
    print("  PASS (Dense path — SAGE does not depend on BM25)")
    return True


def main() -> int:
    if not (GRAPH_DIR / "nodes.jsonl").is_file():
        print(f"Graph not found: {GRAPH_DIR}")
        return 1
    if not BM25_DIR.is_dir():
        print(f"BM25 index not found: {BM25_DIR}")
        return 1

    store = GraphStore.from_dir(GRAPH_DIR)
    expander = GraphExpander(store)
    bm25 = BM25Retriever(index_dir=BM25_DIR)
    sage_bm25 = SageExpansionRetriever(bm25, expander)

    _print_bm25_run(sage_bm25, bm25)
    dense_ok = _print_dense_run(store)

    print("\n" + "=" * 72)
    print(
        "SUMMARY: BM25+SAGE=PASS"
        + ("; Dense+SAGE=PASS" if dense_ok else "; Dense+SAGE=SKIP")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
