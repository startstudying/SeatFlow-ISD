"""Graph tensorization: convert snapshot to PyG HeteroData.

Produces a HeteroData object with node features and edge indices for
each node/edge type. Falls back to plain dicts when PyG is not installed
(e.g. in CI without torch).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from seatflow.graph.snapshot import GraphSnapshot
from seatflow.graph.schema import EdgeType, NodeType


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


def tensorize_snapshot(
    snapshot: GraphSnapshot,
    *,
    feature_dim: int = 16,
) -> TensorizedGraph:
    """Convert a GraphSnapshot to tensorized form.

    Args:
        snapshot: The graph snapshot to tensorize.
        feature_dim: Dimensionality of node features (random init for
            non-semantic features; real features are added later by the
            model).

    Returns:
        TensorizedGraph with node mappings and edge indices.
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
        # Simple deterministic feature from node_id hash
        h = hashlib.sha256(node.node_id.encode("utf-8")).digest()
        feat = [(b / 255.0) for b in h[:feature_dim]]
        node_features[nt].append(feat)

    # Build edge indices
    edge_indices: dict[str, list[tuple[int, int]]] = {}
    edge_features: dict[str, list[list[float]]] = {}

    for edge in snapshot.edges:
        et = edge.edge_type.value
        if et not in edge_indices:
            edge_indices[et] = []
            edge_features[et] = []
        src_type = edge.edge_type.source_type.value if hasattr(edge.edge_type, 'source_type') else None
        # Look up source and target indices
        src_idx = _lookup_node_index(node_id_maps, edge.source_id)
        tgt_idx = _lookup_node_index(node_id_maps, edge.target_id)
        if src_idx is None or tgt_idx is None:
            continue
        edge_indices[et].append((src_idx, tgt_idx))
        # Edge feature from properties
        props = list(edge.properties.values())
        feat = [float(v) if isinstance(v, (int, float)) else 0.0 for v in props[:feature_dim]]
        # Pad to feature_dim
        while len(feat) < feature_dim:
            feat.append(0.0)
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


def _lookup_node_index(
    node_id_maps: dict[str, dict[str, int]], node_id: str
) -> Optional[int]:
    for nt, mapping in node_id_maps.items():
        if node_id in mapping:
            return mapping[node_id]
    return None


def _infer_edge_type_tuple(edge_type_str: str) -> Optional[tuple[str, str, str]]:
    """Map edge type string to (src_type, edge_type, tgt_type) tuple for PyG."""
    try:
        et = EdgeType(edge_type_str)
        schema = et  # EdgeType enum doesn't have source_type/target_type directly
        from seatflow.graph.schema import EDGE_SCHEMAS
        es = EDGE_SCHEMAS[et]
        return (es.source_type.value, et.value, es.target_type.value)
    except (KeyError, ValueError):
        return None
