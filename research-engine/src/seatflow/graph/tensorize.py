"""Graph tensorization: convert snapshot to PyG HeteroData.

Produces a HeteroData object with node features and edge indices for
each node/edge type. Falls back to plain dicts when PyG is not installed
(e.g. in CI without torch).

Only values placed in the explicit ``continuous_features`` property
namespace are eligible for model tensors. Identifiers remain available in
``node_id_maps`` for graph indexing, but never become floating-point input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional

from seatflow.graph.snapshot import GraphSnapshot
from seatflow.graph.schema import (
    EdgeType,
    TensorFeatureSchema,
    get_edge_tensor_feature_schema,
    get_node_tensor_feature_schema,
)


CONTINUOUS_FEATURES_KEY = "continuous_features"

_EXPLICIT_IDENTITY_FEATURES = frozenset({"ts_code"})


@dataclass(frozen=True)
class TensorizedGraph:
    """Result of tensorizing a graph snapshot.

    Stores node ID mappings and edge index arrays. When PyG is available,
    ``hetero_data`` holds the HeteroData object. Otherwise it holds a
    plain dict representation.
    """

    snapshot_id: str
    snapshot_hash: str
    node_id_maps: dict[str, dict[str, int]]  # node_type -> {node_id: index}
    edge_indices: dict[str, list[tuple[int, int]]]  # edge_type -> [(src_idx, tgt_idx)]
    node_features: dict[str, list[list[float]]]  # node_type -> feature matrix
    edge_features: dict[str, list[list[float]]]  # edge_type -> feature matrix
    hetero_data: Any = None  # torch_geometric.data.HeteroData when available


def tensorize_snapshot(snapshot: GraphSnapshot) -> TensorizedGraph:
    """Convert a GraphSnapshot to tensorized form.

    Args:
        snapshot: The graph snapshot to tensorize.
    Returns:
        TensorizedGraph with node mappings and edge indices.

    Raises:
        ValueError: If semantic continuous features are missing, contain an
            identity field, contain a non-finite/non-numeric value, or drift
            within a node/edge type.
    """
    # Build node ID maps
    node_id_maps: dict[str, dict[str, int]] = {}
    node_features: dict[str, list[list[float]]] = {}

    for node in snapshot.nodes:
        nt = node.node_type.value
        if nt not in node_id_maps:
            node_id_maps[nt] = {}
            node_features[nt] = []
        idx = len(node_id_maps[nt])
        node_id_maps[nt][node.node_id] = idx

        feat = _extract_continuous_features(
            owner=f"node {node.node_id!r}",
            properties=node.properties,
            schema=get_node_tensor_feature_schema(node.node_type),
        )
        node_features[nt].append(feat)

    # Build edge indices
    edge_indices: dict[str, list[tuple[int, int]]] = {}
    edge_features: dict[str, list[list[float]]] = {}

    for edge in snapshot.edges:
        et = edge.edge_type.value
        if et not in edge_indices:
            edge_indices[et] = []
            edge_features[et] = []
        src_idx = _lookup_node_index(node_id_maps, edge.source_id)
        tgt_idx = _lookup_node_index(node_id_maps, edge.target_id)
        if src_idx is None or tgt_idx is None:
            continue
        edge_indices[et].append((src_idx, tgt_idx))

        feat = _extract_continuous_features(
            owner=f"edge {edge.edge_id!r}",
            properties=edge.properties,
            schema=get_edge_tensor_feature_schema(edge.edge_type),
        )
        edge_features[et].append(feat)

    # Try to create HeteroData if PyG is available
    hetero_data = None
    try:
        import torch
        from torch_geometric.data import HeteroData

        hetero_data = HeteroData()
        for nt, feats in node_features.items():
            if feats:
                hetero_data[nt].x = torch.tensor(feats, dtype=torch.float32)
        for et, indices in edge_indices.items():
            if indices:
                srcs = [i[0] for i in indices]
                tgts = [i[1] for i in indices]
                # Infer src/tgt node types from edge schema
                edge_type_tuple = _infer_edge_type_tuple(et)
                if edge_type_tuple:
                    hetero_data[edge_type_tuple].edge_index = torch.tensor(
                        [srcs, tgts], dtype=torch.long
                    )
    except ImportError:
        pass

    return TensorizedGraph(
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.snapshot_hash,
        node_id_maps=node_id_maps,
        edge_indices=edge_indices,
        node_features=node_features,
        edge_features=edge_features,
        hetero_data=hetero_data,
    )


def _lookup_node_index(node_id_maps: dict[str, dict[str, int]], node_id: str) -> Optional[int]:
    for mapping in node_id_maps.values():
        if node_id in mapping:
            return mapping[node_id]
    return None


def _extract_continuous_features(
    *,
    owner: str,
    properties: Mapping[str, Any],
    schema: TensorFeatureSchema,
) -> list[float]:
    raw_features = properties.get(CONTINUOUS_FEATURES_KEY)
    if not isinstance(raw_features, Mapping):
        raise ValueError(f"{owner} must provide a {CONTINUOUS_FEATURES_KEY!r} mapping")

    if not all(isinstance(name, str) and name for name in raw_features):
        raise ValueError(f"{owner} continuous feature names must be non-empty strings")

    names = tuple(sorted(raw_features))
    identity_names = [name for name in names if _is_identity_feature(name)]
    if identity_names:
        raise ValueError(
            f"{owner} contains forbidden identity continuous features: {identity_names}"
        )

    expected_names = schema.feature_names
    if set(names) != set(expected_names):
        raise ValueError(
            f"{owner} does not match tensor feature schema {schema.version!r}: "
            f"expected {expected_names}, got {names}"
        )

    values: list[float] = []
    for name in expected_names:
        raw_value = raw_features[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(f"{owner} feature {name!r} must be a real number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{owner} feature {name!r} must be finite")
        values.append(value)

    return values


def _is_identity_feature(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in _EXPLICIT_IDENTITY_FEATURES
        or normalized.endswith("_id")
        or normalized.endswith("_hash")
        or normalized.startswith("hash_")
        or normalized == "hash"
    )


def _infer_edge_type_tuple(edge_type_str: str) -> Optional[tuple[str, str, str]]:
    """Map edge type string to (src_type, edge_type, tgt_type) tuple for PyG."""
    try:
        et = EdgeType(edge_type_str)
        from seatflow.graph.schema import EDGE_SCHEMAS

        es = EDGE_SCHEMAS[et]
        return (es.source_type.value, et.value, es.target_type.value)
    except (KeyError, ValueError):
        return None
