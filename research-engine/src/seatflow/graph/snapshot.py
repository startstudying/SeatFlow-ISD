"""Dynamic Flow Graph snapshot builder.

Constructs a point-in-time graph snapshot from events, participations,
and propagation candidates. The snapshot is immutable once created.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from seatflow.graph.schema import (
    EdgeType,
    NodeType,
    get_edge_schema,
    max_propagation_hops,
    default_top_k,
)


@dataclass(frozen=True)
class GraphNode:
    """A node in the heterogeneous graph."""

    node_id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """An edge in the heterogeneous graph."""

    edge_id: str
    edge_type: EdgeType
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    available_time: Optional[datetime] = None


@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable point-in-time graph snapshot."""

    snapshot_id: str
    cutoff_time: datetime
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_hash:
            object.__setattr__(self, "snapshot_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash over canonical JSON of nodes and edges."""
        data = {
            "snapshot_id": self.snapshot_id,
            "cutoff_time": self.cutoff_time.isoformat(),
            "nodes": sorted(
                [{"id": n.node_id, "type": n.node_type.value, "props": n.properties}
                 for n in self.nodes],
                key=lambda x: x["id"],
            ),
            "edges": sorted(
                [{"id": e.edge_id, "type": e.edge_type.value,
                  "src": e.source_id, "tgt": e.target_id, "props": e.properties}
                 for e in self.edges],
                key=lambda x: x["id"],
            ),
        }
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class GraphSnapshotBuilder:
    """Builder for constructing PIT graph snapshots."""

    def __init__(self, *, top_k: int | None = None, max_hops: int | None = None) -> None:
        self._top_k = top_k or default_top_k()
        self._max_hops = max_hops or max_propagation_hops()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

    def add_stock_node(self, stock_id: str, properties: dict | None = None) -> None:
        self._add_node(stock_id, NodeType.STOCK, properties or {})

    def add_seat_node(self, seat_id: str, properties: dict | None = None) -> None:
        self._add_node(seat_id, NodeType.SEAT, properties or {})

    def add_event_node(self, event_id: str, properties: dict | None = None) -> None:
        self._add_node(event_id, NodeType.EVENT, properties or {})

    def add_theme_node(self, theme_id: str, properties: dict | None = None) -> None:
        self._add_node(theme_id, NodeType.THEME, properties or {})

    def _add_node(self, node_id: str, node_type: NodeType, properties: dict) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = GraphNode(
                node_id=node_id, node_type=node_type, properties=properties
            )

    def add_edge(
        self,
        edge_type: EdgeType,
        source_id: str,
        target_id: str,
        properties: dict | None = None,
        available_time: datetime | None = None,
    ) -> None:
        edge_id = f"{edge_type.value}:{source_id}->{target_id}"
        if edge_id in self._edges:
            return
        self._edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            properties=properties or {},
            available_time=available_time,
        )

    def apply_top_k_truncation(self) -> None:
        """Apply Top-K truncation per edge type per source node."""
        # Group edges by (edge_type, source_id)
        groups: dict[tuple[str, str], list[GraphEdge]] = {}
        for edge in self._edges.values():
            key = (edge.edge_type.value, edge.source_id)
            groups.setdefault(key, []).append(edge)

        # For each group, keep only top_k by candidate_score (or first k)
        kept_edges: dict[str, GraphEdge] = {}
        for (etype, src), edges in groups.items():
            schema = get_edge_schema(EdgeType(etype))
            k = min(self._top_k, schema.top_k)
            # Sort by candidate_score descending, fallback to order
            def sort_key(e: GraphEdge) -> float:
                score = e.properties.get("candidate_score")
                if score is not None:
                    return -float(score)
                return 0.0
            edges.sort(key=sort_key)
            for edge in edges[:k]:
                kept_edges[edge.edge_id] = edge

        self._edges = kept_edges

    def build(self, *, snapshot_id: str, cutoff_time: datetime) -> GraphSnapshot:
        """Build the immutable snapshot."""
        nodes = list(self._nodes.values())
        edges = list(self._edges.values())
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        for n in nodes:
            node_counts[n.node_type.value] = node_counts.get(n.node_type.value, 0) + 1
        for e in edges:
            edge_counts[e.edge_type.value] = edge_counts.get(e.edge_type.value, 0) + 1
        return GraphSnapshot(
            snapshot_id=snapshot_id,
            cutoff_time=cutoff_time,
            nodes=nodes,
            edges=edges,
            node_counts=node_counts,
            edge_counts=edge_counts,
        )

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
