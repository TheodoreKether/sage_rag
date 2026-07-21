"""Load Evidence Units and persist Standard Evidence Graph artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

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


class GraphStore:
    """JSONL persistence for nodes / edges under ``data/sage_graph/``."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    @property
    def nodes_path(self) -> Path:
        return self.output_dir / "nodes.jsonl"

    @property
    def edges_path(self) -> Path:
        return self.output_dir / "edges.jsonl"

    @property
    def statistics_path(self) -> Path:
        return self.output_dir / "graph_statistics.md"

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
