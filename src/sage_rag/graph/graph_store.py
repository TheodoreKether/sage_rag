"""Load / persist Standard Evidence Graph and serve in-memory lookups.

Supports both:
- write path used by ``build_graph.py`` (``save`` / ``load``)
- read path used by expansion / future retrievers (``get_node`` / ``get_neighbors``)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .graph_schema import Edge, GraphStatistics, Node

logger = logging.getLogger(__name__)


def load_evidence_units(path: Path) -> list[dict[str, Any]]:
    """Load Evidence Units from a JSONL file (one unit per line)."""
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                units.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skip invalid JSONL at %s:%d (%s)", path, line_no, exc)
    return units


def write_statistics_markdown(stats: GraphStatistics, path: Path) -> None:
    """Write human-readable graph statistics markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nc = stats.node_counts
    ec = stats.edge_counts
    lines = [
        "# Standard Evidence Graph Statistics",
        "",
        "## Nodes",
        "",
        "| Type | Count |",
        "|------|------:|",
        f"| document | {nc.get('document', 0)} |",
        f"| chapter | {nc.get('chapter', 0)} |",
        f"| clause | {nc.get('clause', 0)} |",
        f"| evidence | {nc.get('evidence', 0)} |",
        f"| **total** | **{sum(nc.values())}** |",
        "",
        "## Edges",
        "",
        "| Type | Count |",
        "|------|------:|",
        f"| parent_of | {ec.get('parent_of', 0)} |",
        f"| next_to | {ec.get('next_to', 0)} |",
        f"| refers_to | {ec.get('refers_to', 0)} |",
        f"| **total** | **{sum(ec.values())}** |",
        "",
        "## refers_to resolution",
        "",
        f"- resolved: {stats.refers_to_resolved}",
        f"- unresolved (target kept as reference string): {stats.refers_to_unresolved}",
        "",
        "## Notes",
        "",
        "- Hierarchy: `document --parent_of--> chapter --parent_of--> clause --parent_of--> evidence`",
        "- Node ids: `document_id`, `{doc}::chapter::{chapter_id}`, `{doc}::clause::{parent_clause}`, `unit_id`",
        "- `next_to`: adjacent clauses under the same `(document_id, chapter_id)`, weight=0.3",
        "- `refers_to`: rule-based cues (`见` / `参见` / `附录` / `GB/T` / `ISO`), weight=0.5",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote statistics to %s", path)


@dataclass(frozen=True)
class Neighbor:
    """One hop from a node, including direction relative to the query node."""

    node_id: str
    edge_type: str
    direction: str  # "out" (query→neighbor) | "in" (neighbor→query)
    weight: float
    edge: Edge
    node: Node | None = None  # None when target is an unresolved reference string


class GraphStore:
    """JSONL persistence + in-memory adjacency indexes for the Standard Evidence Graph."""

    def __init__(self, graph_dir: Path | str, *, auto_load: bool = True) -> None:
        self.output_dir = Path(graph_dir)
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        if auto_load and self.nodes_path.is_file() and self.edges_path.is_file():
            nodes, edges = self.load()
            self.build_indexes(nodes, edges)

    # ------------------------------------------------------------------ paths

    @property
    def nodes_path(self) -> Path:
        return self.output_dir / "nodes.jsonl"

    @property
    def edges_path(self) -> Path:
        return self.output_dir / "edges.jsonl"

    @property
    def statistics_path(self) -> Path:
        return self.output_dir / "graph_statistics.md"

    # ------------------------------------------------------------------ index

    def build_indexes(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        """Replace in-memory indexes from node/edge lists."""
        self._nodes = {n.id: n for n in nodes}
        self._edges = list(edges)
        self._out = defaultdict(list)
        self._in = defaultdict(list)
        for edge in self._edges:
            self._out[edge.source].append(edge)
            self._in[edge.target].append(edge)
        logger.info(
            "GraphStore indexed: %d nodes, %d edges (dir=%s)",
            len(self._nodes),
            len(self._edges),
            self.output_dir,
        )

    @classmethod
    def from_dir(cls, graph_dir: Path | str) -> "GraphStore":
        """Load ``nodes.jsonl`` / ``edges.jsonl`` and build indexes."""
        return cls(graph_dir, auto_load=True)

    # ------------------------------------------------------------------ query

    def get_node(self, node_id: str) -> Node | None:
        """Return a node by id, or ``None`` if missing / unresolved reference."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def iter_nodes(self, node_type: str | None = None) -> list[Node]:
        if node_type is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.type == node_type]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Sequence[str] | None = None,
        *,
        direction: str = "both",
    ) -> list[Neighbor]:
        """Return 1-hop neighbors.

        Parameters
        ----------
        node_id:
            Query node id (must exist for meaningful results; unresolved ids
            may still appear as edge endpoints).
        edge_types:
            Optional allow-list, e.g. ``["parent_of"]``. ``None`` = all types.
        direction:
            ``"out"`` — edges where ``source == node_id``;
            ``"in"`` — edges where ``target == node_id``;
            ``"both"`` — union (default).
        """
        allowed = set(edge_types) if edge_types is not None else None
        results: list[Neighbor] = []

        if direction in ("out", "both"):
            for edge in self._out.get(node_id, []):
                if allowed is not None and edge.type not in allowed:
                    continue
                results.append(
                    Neighbor(
                        node_id=edge.target,
                        edge_type=edge.type,
                        direction="out",
                        weight=edge.weight,
                        edge=edge,
                        node=self._nodes.get(edge.target),
                    )
                )

        if direction in ("in", "both"):
            for edge in self._in.get(node_id, []):
                if allowed is not None and edge.type not in allowed:
                    continue
                results.append(
                    Neighbor(
                        node_id=edge.source,
                        edge_type=edge.type,
                        direction="in",
                        weight=edge.weight,
                        edge=edge,
                        node=self._nodes.get(edge.source),
                    )
                )

        return results

    def get_evidence_children(self, node_id: str) -> list[Node]:
        """Outgoing ``parent_of`` children that are evidence nodes."""
        out: list[Node] = []
        for nb in self.get_neighbors(node_id, edge_types=["parent_of"], direction="out"):
            if nb.node is not None and nb.node.type == "evidence":
                out.append(nb.node)
        return out

    def get_clause_children(self, node_id: str) -> list[Node]:
        """Outgoing ``parent_of`` children that are clause nodes."""
        out: list[Node] = []
        for nb in self.get_neighbors(node_id, edge_types=["parent_of"], direction="out"):
            if nb.node is not None and nb.node.type == "clause":
                out.append(nb.node)
        return out

    # ------------------------------------------------------------------ I/O

    def save(
        self,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        stats: GraphStatistics | None = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        node_list = list(nodes)
        edge_list = list(edges)

        with self.nodes_path.open("w", encoding="utf-8") as fh:
            for node in node_list:
                fh.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

        with self.edges_path.open("w", encoding="utf-8") as fh:
            for edge in edge_list:
                fh.write(json.dumps(edge.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            "Saved graph: %d nodes → %s, %d edges → %s",
            len(node_list),
            self.nodes_path,
            len(edge_list),
            self.edges_path,
        )

        if stats is not None:
            write_statistics_markdown(stats, self.statistics_path)

        self.build_indexes(node_list, edge_list)

    def load(self) -> tuple[list[Node], list[Edge]]:
        nodes: list[Node] = []
        edges: list[Edge] = []
        with self.nodes_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                nodes.append(
                    Node(id=obj["id"], type=obj["type"], attributes=obj.get("attributes") or {})
                )
        with self.edges_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                edges.append(
                    Edge(
                        source=obj["source"],
                        target=obj["target"],
                        type=obj["type"],
                        weight=float(obj.get("weight", 1.0)),
                        attributes=obj.get("attributes") or {},
                    )
                )
        return nodes, edges

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)
