"""Graph relation masking helpers with strict non-mutating semantics."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Literal

from seatflow.graph.tensorize import TensorizedGraph
from seatflow.graph.types import GraphEdge, HeteroGraphSnapshot
from seatflow.graph.types import GraphEdge as EdgeType


def relation_types(snapshot: HeteroGraphSnapshot) -> tuple[str, ...]:
    return tuple(sorted({edge.relation for edge in snapshot.edges}))


def mask_relations(
    snapshot: HeteroGraphSnapshot,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> HeteroGraphSnapshot:
    """
    Return a cloned HeteroGraphSnapshot with edge relations filtered.

    The original snapshot is not modified.
    """
    include_set = frozenset(include or ())
    exclude_set = frozenset(exclude or ())

    if include is not None and not include_set:
        kept_edges: tuple[EdgeType, ...] = ()
    elif include is not None:
        kept_edges = tuple(
            _clone_edge(edge)
            for edge in snapshot.edges
            if edge.relation in include_set
        )
    else:
        kept_edges = tuple(
            _clone_edge(edge)
            for edge in snapshot.edges
            if edge.relation not in exclude_set
        )

    return HeteroGraphSnapshot(
        as_of_date=snapshot.as_of_date,
        nodes=tuple(snapshot.nodes),
        edges=kept_edges,
    )


def _clone_edge(edge: GraphEdge) -> GraphEdge:
    return GraphEdge(
        source_type=edge.source_type,
        source_id=edge.source_id,
        relation=edge.relation,
        target_type=edge.target_type,
        target_id=edge.target_id,
        weight=edge.weight,
        event_date=edge.event_date,
        attributes=dict(edge.attributes),
    )


RelationMaskMode = Literal[
    "full",
    "without_seat_stock",
    "without_industry_stock",
    "without_stock_stock",
    "stock_only",
]


def mask_tensorized_relations(
    graph: TensorizedGraph,
    mode: RelationMaskMode,
) -> TensorizedGraph:
    """Clone a tensorized graph and empty relations selected by an audit mode."""
    cloned_data = graph.data.clone()
    metadata = deepcopy(graph.metadata)
    if mode == "full":
        return TensorizedGraph(data=cloned_data, metadata=metadata)
    for edge_type in cloned_data.edge_types:
        if not _should_mask(edge_type, mode):
            continue
        store = cloned_data[edge_type]
        store.edge_index = store.edge_index[:, :0].clone()
        store.edge_attr = store.edge_attr[:0].clone()
        if hasattr(store, "edge_weight"):
            store.edge_weight = store.edge_weight[:0].clone()
        edge_key = "::".join(edge_type)
        edge_shapes = metadata.get("edge_shapes")
        if isinstance(edge_shapes, dict):
            edge_shapes[edge_key] = list(store.edge_index.shape)
        attr_shapes = metadata.get("edge_attr_shapes")
        if isinstance(attr_shapes, dict):
            attr_shapes[edge_key] = list(store.edge_attr.shape)
    return TensorizedGraph(data=cloned_data, metadata=metadata)


def _should_mask(
    edge_type: tuple[str, str, str],
    mode: RelationMaskMode,
) -> bool:
    source, _, target = edge_type
    endpoints = {source, target}
    if mode == "stock_only":
        return True
    if mode == "without_seat_stock":
        return endpoints == {"seat", "stock"}
    if mode == "without_industry_stock":
        return endpoints == {"industry", "stock"}
    if mode == "without_stock_stock":
        return source == target == "stock"
    raise ValueError(f"unsupported relation mask mode: {mode}")


__all__ = [
    "RelationMaskMode",
    "mask_relations",
    "mask_tensorized_relations",
    "relation_types",
]
