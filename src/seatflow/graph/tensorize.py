from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import copysign, log1p
from typing import Any

from seatflow.feature_registry import FeatureKind, infer_feature_kind
from seatflow.graph.edge_schema import transform_edge_value, validate_edge_attributes
from seatflow.graph.types import GraphEdge, GraphNode, HeteroGraphSnapshot, NodeType


class PyGUnavailableError(RuntimeError):
    pass


TENSORIZATION_VERSION = "seatflow_tensor_v3_no_id_hash"
_LOG1P_NODE_FEATURES = frozenset({"seat_profile_event_count", "stock_bar_count"})
_LOG1P_EDGE_ATTRIBUTES = frozenset(
    {
        "amount",
        "edge_gate_input",
        "graph_edge_weight",
        "source_amount",
        "target_amount",
    }
)
_PRETRANSFORMED_EDGE_ATTRIBUTES = frozenset({"signed_log_amount_sum"})


def is_pretransformed_feature(name: str) -> bool:
    from seatflow.graph.edge_schema import EdgeFeatureType, classify_edge_field

    return (
        name in _LOG1P_NODE_FEATURES
        or name in _LOG1P_EDGE_ATTRIBUTES
        or name in _PRETRANSFORMED_EDGE_ATTRIBUTES
        or classify_edge_field(name).kind == EdgeFeatureType.COUNT
    )


@dataclass(frozen=True)
class TensorizedGraph:
    data: Any
    metadata: dict[str, object]


@dataclass(frozen=True)
class FrozenGraphSchema:
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    node_feature_schema: dict[str, tuple[str, ...]]
    edge_attr_schema: dict[tuple[str, str, str], tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_types": list(self.node_types),
            "edge_types": [list(edge_type) for edge_type in self.edge_types],
            "node_feature_schema": {
                node_type: list(schema)
                for node_type, schema in sorted(self.node_feature_schema.items())
            },
            "edge_attr_schema": {
                "::".join(edge_type): list(schema)
                for edge_type, schema in sorted(self.edge_attr_schema.items())
            },
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def derive_frozen_graph_schema(
    snapshots: Iterable[HeteroGraphSnapshot],
) -> FrozenGraphSchema:
    materialized = tuple(snapshots)
    node_feature_keys: dict[str, set[str]] = {node_type.value: set() for node_type in NodeType}
    edge_attr_keys: dict[tuple[str, str, str], set[str]] = {}
    for snapshot in materialized:
        for node_type in NodeType:
            node_feature_keys[node_type.value].update(
                _node_feature_keys(snapshot.nodes_of_type(node_type))
            )
        for edge_type, edges in _edges_by_type(snapshot).items():
            edge_attr_keys.setdefault(edge_type, set()).update(_edge_attr_keys(edges))
    return FrozenGraphSchema(
        node_types=tuple(node_type.value for node_type in NodeType),
        edge_types=tuple(sorted(edge_attr_keys)),
        node_feature_schema={
            node_type: tuple(sorted(keys)) or ("bias",)
            for node_type, keys in node_feature_keys.items()
        },
        edge_attr_schema={
            edge_type: tuple(sorted(keys | {"edge_gate_input", "graph_edge_weight"}))
            for edge_type, keys in edge_attr_keys.items()
        },
    )


def tensorize_hetero_graphs(
    snapshots: Iterable[HeteroGraphSnapshot],
    *,
    frozen_schema: FrozenGraphSchema | None = None,
) -> tuple[TensorizedGraph, ...]:
    materialized = tuple(snapshots)
    schema = frozen_schema or derive_frozen_graph_schema(materialized)
    return tuple(
        tensorize_hetero_graph(snapshot, frozen_schema=schema)
        for snapshot in materialized
    )


def tensorize_hetero_graph(
    snapshot: HeteroGraphSnapshot,
    *,
    frozen_schema: FrozenGraphSchema | None = None,
) -> TensorizedGraph:
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ModuleNotFoundError as error:
        raise PyGUnavailableError(
            "PyTorch and torch-geometric are required for HeteroData tensorization. "
            "Install the ml extra: python -m pip install -e .[ml]"
        ) from error

    schema = frozen_schema or derive_frozen_graph_schema((snapshot,))
    _validate_snapshot_matches_schema(snapshot, schema)
    data = HeteroData()
    node_mapping: dict[str, dict[str, int]] = {}
    feature_schema: dict[str, list[str]] = {}
    for node_type in NodeType:
        nodes = sorted(snapshot.nodes_of_type(node_type), key=lambda item: item.node_id)
        node_schema = list(schema.node_feature_schema[node_type.value])
        feature_schema[node_type.value] = node_schema
        node_mapping[node_type.value] = {node.node_id: index for index, node in enumerate(nodes)}
        width = max(1, len(node_schema))
        features = [_node_features(node.features, node_schema) for node in nodes]
        data[node_type.value].x = (
            torch.tensor(features, dtype=torch.float32).reshape((len(nodes), width))
            if features
            else torch.zeros((0, width), dtype=torch.float32)
        )

    edge_shapes: dict[str, list[int]] = {}
    edge_attr_schema: dict[str, list[str]] = {}
    grouped_edges = _edges_by_type(snapshot)
    for edge_type in schema.edge_types:
        edges = grouped_edges.get(edge_type, [])
        source_type, relation, target_type = edge_type
        source_mapping = node_mapping[source_type]
        target_mapping = node_mapping[target_type]
        edge_key = "::".join(edge_type)
        attr_schema = list(schema.edge_attr_schema[edge_type])
        edge_attr_schema[edge_key] = attr_schema
        pairs: list[list[int]] = []
        attributes: list[list[float]] = []
        for edge in edges:
            if edge.source_id not in source_mapping or edge.target_id not in target_mapping:
                raise ValueError("edge references a missing node")
            pairs.append([source_mapping[edge.source_id], target_mapping[edge.target_id]])
            attributes.append(_edge_features(edge, attr_schema))
        edge_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.empty((2, 0), dtype=torch.long)
        )
        attr_width = max(1, len(attr_schema))
        edge_attr = (
            torch.tensor(attributes, dtype=torch.float32).reshape((len(edges), attr_width))
            if attributes
            else torch.zeros((0, attr_width), dtype=torch.float32)
        )
        data[(source_type, relation, target_type)].edge_index = edge_index
        data[(source_type, relation, target_type)].edge_attr = edge_attr
        data[(source_type, relation, target_type)].edge_weight = _edge_gate_weight(
            edge_attr, attr_schema
        )
        edge_shapes[edge_key] = list(edge_index.shape)

    metadata: dict[str, object] = {
        "as_of_date": snapshot.as_of_date.isoformat(),
        "tensorization_version": TENSORIZATION_VERSION,
        "tensor_schema_version": TENSORIZATION_VERSION,
        "edge_feature_schema_hash": schema.fingerprint(),
        "node_mapping": node_mapping,
        "coverage_audit": _static_node_coverage_audit(node_mapping),
        "feature_schema": feature_schema,
        "frozen_graph_schema": schema.to_dict(),
        "dtype": "float32",
        "node_shapes": {
            node_type.value: list(data[node_type.value].x.shape) for node_type in NodeType
        },
        "edge_shapes": edge_shapes,
        "edge_attr_schema": edge_attr_schema,
        "pretransformed_features": {
            "node": sorted(_LOG1P_NODE_FEATURES),
            "edge": sorted(_LOG1P_EDGE_ATTRIBUTES),
        },
        "edge_attr_shapes": {
            "::".join(edge_type): list(data[edge_type].edge_attr.shape)
            for edge_type in data.edge_types
        },
    }
    return TensorizedGraph(data=data, metadata=metadata)


def _static_node_coverage_audit(
    node_mapping: dict[str, dict[str, int]],
) -> dict[str, object]:
    missing = [
        node_type.value
        for node_type in (NodeType.INDUSTRY, NodeType.CONCEPT)
        if not node_mapping[node_type.value]
    ]
    return {
        "industry_node_count": len(node_mapping[NodeType.INDUSTRY.value]),
        "concept_node_count": len(node_mapping[NodeType.CONCEPT.value]),
        "missing_static_node_types": missing,
    }


def _node_feature_keys(nodes: Sequence[GraphNode]) -> set[str]:
    keys = sorted(
        {
            key
            for node in nodes
            for key, value in node.features.items()
            if _is_numeric(value)
        }
    )
    _reject_nonnumeric_values("node feature", (node.features for node in nodes))
    _reject_entity_ids("node feature", keys)
    return set(keys)


def _node_features(features: dict[str, Any], schema: list[str]) -> list[float]:
    if schema == ["bias"]:
        return [1.0]
    return [_scaled_node_value(key, float(features.get(key, 0.0))) for key in schema]


def _edge_attr_keys(edges: Sequence[GraphEdge]) -> set[str]:
    for edge in edges:
        validate_edge_attributes(edge.attributes)
    keys = sorted(
        {
            key
            for edge in edges
            for key, value in edge.attributes.items()
            if _is_numeric(value)
        }
    )
    _reject_nonnumeric_values("edge attribute", (edge.attributes for edge in edges))
    _reject_entity_ids("edge attribute", keys)
    return set(keys)


def _edge_features(edge: Any, schema: list[str]) -> list[float]:
    values = dict(edge.attributes)
    values.setdefault("graph_edge_weight", float(edge.weight))
    values.setdefault("edge_gate_input", float(edge.weight))
    return [_scaled_edge_value(key, float(values.get(key, 0.0))) for key in schema]


def _scaled_node_value(name: str, value: float) -> float:
    if name in _LOG1P_NODE_FEATURES:
        return _signed_log1p(value)
    return value


def _scaled_edge_value(name: str, value: float) -> float:
    if name in _LOG1P_EDGE_ATTRIBUTES:
        return _signed_log1p(value)
    return transform_edge_value(name, value)


def _signed_log1p(value: float) -> float:
    return copysign(log1p(abs(value)), value)


def _edge_gate_weight(edge_attr: Any, schema: list[str]) -> Any:
    index = schema.index("edge_gate_input") if "edge_gate_input" in schema else 0
    return edge_attr[:, index].contiguous()


def _validate_snapshot_matches_schema(
    snapshot: HeteroGraphSnapshot,
    schema: FrozenGraphSchema,
) -> None:
    node_types = {node_type.value for node_type in NodeType}
    if set(schema.node_types) != node_types:
        raise ValueError("frozen graph schema node types conflict with NodeType enum")
    if set(schema.node_feature_schema) != node_types:
        raise ValueError("frozen graph schema node feature schema is incomplete")
    schema_edge_types = set(schema.edge_types)
    if set(schema.edge_attr_schema) != schema_edge_types:
        raise ValueError("frozen graph schema edge attr schema conflicts with edge types")
    for node_type in NodeType:
        actual = _node_feature_keys(snapshot.nodes_of_type(node_type))
        allowed = set(schema.node_feature_schema.get(node_type.value, ()))
        unexpected = actual - allowed
        if unexpected:
            raise ValueError(
                f"schema conflict for node type {node_type.value}: "
                f"unexpected feature keys {sorted(unexpected)}"
            )
    grouped = _edges_by_type(snapshot)
    unexpected_edge_types = set(grouped) - schema_edge_types
    if unexpected_edge_types:
        raise ValueError(
            "schema conflict for edge types: "
            f"unexpected edge types {sorted('::'.join(item) for item in unexpected_edge_types)}"
        )
    for edge_type, edges in grouped.items():
        actual = _edge_attr_keys(edges)
        allowed = set(schema.edge_attr_schema.get(edge_type, ()))
        unexpected = actual - allowed
        if unexpected:
            raise ValueError(
                f"schema conflict for edge type {'::'.join(edge_type)}: "
                f"unexpected attribute keys {sorted(unexpected)}"
            )


def _reject_nonnumeric_values(
    context: str,
    records: Iterable[dict[str, Any]],
) -> None:
    for record in records:
        for key, value in record.items():
            if not _is_numeric(value):
                raise ValueError(f"schema conflict: {context} {key!r} is not numeric")


def _reject_entity_ids(context: str, keys: Iterable[str]) -> None:
    invalid = sorted(key for key in keys if infer_feature_kind(key) == FeatureKind.ENTITY_ID)
    if invalid:
        raise ValueError(
            f"schema conflict: {context} contains entity identifiers "
            f"in continuous features: {invalid}"
        )


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _edges_by_type(
    snapshot: HeteroGraphSnapshot,
) -> dict[tuple[str, str, str], list[Any]]:
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    for edge in sorted(
        snapshot.edges,
        key=lambda item: (
            item.event_date,
            item.source_type.value,
            item.source_id,
            item.relation,
            item.target_type.value,
            item.target_id,
        ),
    ):
        key = (edge.source_type.value, edge.relation, edge.target_type.value)
        grouped.setdefault(key, []).append(edge)
    return grouped
