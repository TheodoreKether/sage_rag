"""Manual / CLI tests for GraphStore + GraphExpander.

Usage:
  python src/sage_rag/test_graph_expansion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sage_rag.expansion.graph_expander import ExpandedEvidence, GraphExpander
from src.sage_rag.graph.graph_store import GraphStore

GRAPH_DIR = _REPO_ROOT / "data" / "sage_graph"

CASE1_UNIT = "GB_T_39412-2020::6::6.1.1.13::1"
CASE2_UNIT = "IEC_62771-2012::1::1::1"  # resolved refers_to → Annex A


def _preview(text: str, n: int = 100) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def _print_case(title: str, seed_id: str, hits: list[ExpandedEvidence], store: GraphStore) -> None:
    seed = store.get_node(seed_id)
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Original Evidence:")
    print(f"  unit_id: {seed_id}")
    if seed is not None:
        print(f"  text: {_preview(str((seed.attributes or {}).get('text') or ''), 160)}")
    print()
    print(f"Expanded Evidence ({len(hits)}):")
    if not hits:
        print("  (none)")
        return
    for i, hit in enumerate(hits, start=1):
        print(f"{i}.")
        print(f"  unit_id:  {hit.unit_id}")
        print(f"  relation: {hit.relation}")
        if hit.via_node_id:
            print(f"  via:      {hit.via_node_id}")
        chapter = (hit.attributes or {}).get("parent_chapter_id")
        parent_clause = (hit.attributes or {}).get("parent_clause_id")
        if parent_clause:
            print(f"  parent_clause: {parent_clause}")
        if chapter:
            print(f"  parent_chapter: {chapter}")
        ref = (hit.attributes or {}).get("reference")
        if ref:
            print(f"  reference: {ref}")
        print(f"  text: {_preview(hit.text, 100)}")
        print()


def run_case1(expander: GraphExpander, store: GraphStore) -> bool:
    hits = expander.expand(CASE1_UNIT, depth=1)
    _print_case("Case 1: hierarchy + next_to", CASE1_UNIT, hits, store)

    parent_ids = {h.unit_id for h in hits if h.relation == "parent_of"}
    next_ids = {h.unit_id for h in hits if h.relation == "next_to"}
    ok_parent = "GB_T_39412-2020::6::6.1.1::1" in parent_ids
    ok_prev = "GB_T_39412-2020::6::6.1.1.12::1" in next_ids
    ok_next = "GB_T_39412-2020::6::6.1.1.14::1" in next_ids
    chapter_ok = any(
        (h.attributes or {}).get("parent_chapter_id") == "GB_T_39412-2020::chapter::6"
        for h in hits
        if h.relation == "parent_of"
    )
    # refers_to may be empty for this unit (B.11 not linked in v1 graph)
    print("Checks:")
    print(f"  parent clause 6.1.1 evidence: {'PASS' if ok_parent else 'FAIL'}")
    print(f"  parent chapter 6 metadata:    {'PASS' if chapter_ok else 'FAIL'}")
    print(f"  next_to 6.1.1.12:             {'PASS' if ok_prev else 'FAIL'}")
    print(f"  next_to 6.1.1.14:             {'PASS' if ok_next else 'FAIL'}")
    print()
    return ok_parent and ok_prev and ok_next and chapter_ok


def run_case2(expander: GraphExpander, store: GraphStore) -> bool:
    hits = expander.expand(CASE2_UNIT, depth=1)
    _print_case("Case 2: refers_to → Annex A", CASE2_UNIT, hits, store)

    refers = [h for h in hits if h.relation == "refers_to"]
    via_annex = any(
        h.via_node_id == "IEC_62771-2012::chapter::Annex A" for h in refers
    )
    print("Checks:")
    print(f"  refers_to hits > 0:           {'PASS' if refers else 'FAIL'}")
    print(f"  via Annex A chapter:          {'PASS' if via_annex else 'FAIL'}")
    print()
    return bool(refers) and via_annex


def main() -> int:
    if not (GRAPH_DIR / "nodes.jsonl").is_file():
        print(f"Graph not found under {GRAPH_DIR}. Run build_graph.py first.")
        return 1

    store = GraphStore.from_dir(GRAPH_DIR)
    print(
        f"Loaded GraphStore: {store.num_nodes} nodes, {store.num_edges} edges "
        f"from {GRAPH_DIR}"
    )
    # Smoke: get_node / get_neighbors
    node = store.get_node(CASE1_UNIT)
    nbs = store.get_neighbors(
        "GB_T_39412-2020::clause::6.1.1.13", edge_types=["next_to", "parent_of"]
    )
    print(f"get_node({CASE1_UNIT!r}) -> type={node.type if node else None}")
    print(
        f"get_neighbors(clause::6.1.1.13, parent_of|next_to) -> {len(nbs)} neighbors"
    )
    print()

    expander = GraphExpander(store)
    ok1 = run_case1(expander, store)
    ok2 = run_case2(expander, store)

    print("=" * 72)
    print(f"SUMMARY: Case1={'PASS' if ok1 else 'FAIL'}  Case2={'PASS' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2) else 2


if __name__ == "__main__":
    raise SystemExit(main())
