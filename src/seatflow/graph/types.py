from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    SEAT = "seat"
    STOCK = "stock"
    INDUSTRY = "industry"
    CONCEPT = "concept"
    EVENT_REASON = "event_reason"


@dataclass(frozen=True)
class GraphNode:
    node_type: NodeType
    node_id: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source_type: NodeType
    source_id: str
    relation: str
    target_type: NodeType
    target_id: str
    weight: float
    event_date: date
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HeteroGraphSnapshot:
    as_of_date: date
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def nodes_of_type(self, node_type: NodeType) -> tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if node.node_type == node_type)
