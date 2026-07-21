"""Schema for the Standard Evidence Graph (lightweight heterogeneous graph).

G = (V, E) with node types {document, chapter, clause, evidence}
and structural edge types {parent_of, next_to, refers_to}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DOCUMENT = "document"
    CHAPTER = "chapter"
    CLAUSE = "clause"
    EVIDENCE = "evidence"


class EdgeType(str, Enum):
    PARENT_OF = "parent_of"
    NEXT_TO = "next_to"
    REFERS_TO = "refers_to"


# Default edge weights for v1 structural graph.
PARENT_OF_WEIGHT = 1.0
NEXT_TO_WEIGHT = 0.3
REFERS_TO_WEIGHT = 0.5


@dataclass
class Node:
    """Heterogeneous graph node.

    ``id`` conventions (unit_id / document_id never rewritten):
    - document: ``document_id``
    - chapter:  ``document_id::chapter::{chapter_id}``
    - clause:   ``document_id::clause::{parent_clause}``
    - evidence: ``unit_id``
    """

    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "attributes": self.attributes}


# Node id conventions (v1):
#   document: ``{document_id}``
#   chapter:  ``{document_id}::chapter::{chapter_id}``
#   clause:   ``{document_id}::clause::{parent_clause}``
#   evidence: ``{unit_id}`` (unchanged)
# Typed namespaces avoid collisions when chapter_id == parent_clause (e.g. both ``1``).


@dataclass
class Edge:
    """Directed edge. ``target`` may be a node id or an unresolved reference string."""

    source: str
    target: str
    type: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
        }
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    def key(self) -> tuple[str, str, str]:
        """Dedup key: (source, target, type)."""
        return (self.source, self.target, self.type)


@dataclass
class GraphStatistics:
    """Aggregate counts for reporting."""

    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    num_documents: int = 0
    num_evidence_units: int = 0
    refers_to_resolved: int = 0
    refers_to_unresolved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
