"""Fold-local graph-tensor transforms with explicit train/test isolation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, cast

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.feature_registry import FeatureKind, infer_feature_kind
from seatflow.feature_transforms import (
    FeatureTransformError,
    FeatureTransformState,
    apply_feature_transform,
    feature_transform_diagnostics,
    fit_feature_transform,
)
from seatflow.graph.tensorize import TensorizedGraph, is_pretransformed_feature

GRAPH_TRANSFORM_STATE_VERSION = "seatflow_fold_graph_transform_v1"


class GraphTransformError(ValueError):
    """Raised when graph tensors cannot safely use a fold transform."""


@dataclass(frozen=True)
class GraphFeatureTransformState:
    """One fitted numeric state per node/edge tensor block."""

    blocks: dict[str, FeatureTransformState]
    train_manifest_hash: str
    version: str = GRAPH_TRANSFORM_STATE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "train_manifest_hash": self.train_manifest_hash,
            "blocks": {name: state.to_dict() for name, state in sorted(self.blocks.items())},
        }

    def fingerprint(self) -> str:
        return _hash_json(self.to_dict())

    def validate(self) -> None:
        if self.version != GRAPH_TRANSFORM_STATE_VERSION:
            raise GraphTransformError(f"unsupported graph transform version: {self.version!r}")
        _validate_hash(self.train_manifest_hash, "train manifest hash")
        if not self.blocks:
            raise GraphTransformError(
                "graph transform must contain at least one fitted tensor block"
            )
        for block, state in self.blocks.items():
            if not block.startswith(("node::", "edge::")):
                raise GraphTransformError(f"invalid graph transform block: {block!r}")
            state.validate()


@dataclass(frozen=True)
class FoldGraphTransform:
    train: tuple[TensorizedWindowSample, ...]
    test: tuple[TensorizedWindowSample, ...]
    state: GraphFeatureTransformState
    diagnostics: dict[str, object]


def fit_apply_fold_graph_transform(
    train: Sequence[TensorizedWindowSample], test: Sequence[TensorizedWindowSample]
) -> FoldGraphTransform:
    """Fit from train graphs only, then clone-and-apply to both partitions."""
    if not train or not test:
        raise GraphTransformError(
            "fold graph transforms require non-empty train and test partitions"
        )
    train_graphs = _unique_graphs(train)
    test_graphs = _unique_graphs(test)
    if set(train_graphs) & set(test_graphs):
        raise GraphTransformError(
            "a graph object cannot be shared across train and test partitions"
        )
    manifest_hash = _hash_json(
        [
            {"sample_id": sample.sample_id, "source_hash": sample.source_hash}
            for sample in sorted(train, key=lambda item: item.sample_id)
        ]
    )
    blocks = _fit_blocks(train_graphs, manifest_hash)
    state = GraphFeatureTransformState(blocks=blocks, train_manifest_hash=manifest_hash)
    state.validate()
    transformed_train, train_rows = _apply_partition(train, state)
    transformed_test, test_rows = _apply_partition(test, state)
    diagnostics = {
        "schema_version": 1,
        "transform_state_hash": state.fingerprint(),
        "train": _partition_diagnostics(train_rows, state),
        "test": _partition_diagnostics(test_rows, state),
    }
    diagnostics["diagnostics_hash"] = _hash_json(diagnostics)
    return FoldGraphTransform(
        train=transformed_train, test=transformed_test, state=state, diagnostics=diagnostics
    )


def _fit_blocks(
    graphs: Mapping[int, TensorizedGraph], manifest_hash: str
) -> dict[str, FeatureTransformState]:
    blocks: dict[str, FeatureTransformState] = {}
    schemas = _block_schemas(tuple(graphs.values()))
    for block, names in schemas.items():
        rows: list[dict[str, float]] = []
        dates: list[str] = []
        sample_ids: list[str] = []
        for graph in graphs.values():
            graph_rows = _rows_for_block(graph, block, names)
            as_of = str(graph.metadata.get("as_of_date", ""))
            identity = _graph_identity(graph)
            for index, row in enumerate(graph_rows):
                rows.append(row)
                dates.append(as_of)
                sample_ids.append(f"{identity}:{block}:{index}")
        if rows:
            try:
                blocks[block] = fit_feature_transform(
                    rows,
                    names,
                    sample_ids=sample_ids,
                    manifest_hash=manifest_hash,
                    dates=dates,
                    pretransformed_features=tuple(
                        name for name in names if is_pretransformed_feature(name)
                    ),
                )
            except FeatureTransformError as error:
                raise GraphTransformError(f"cannot fit graph tensor block {block}") from error
    if not blocks:
        raise GraphTransformError("training graphs have no continuous tensor features to transform")
    return blocks


def _apply_partition(
    samples: Sequence[TensorizedWindowSample], state: GraphFeatureTransformState
) -> tuple[tuple[TensorizedWindowSample, ...], dict[str, list[dict[str, float]]]]:
    replacements: dict[int, TensorizedGraph] = {}
    rows_by_block: dict[str, list[dict[str, float]]] = {block: [] for block in state.blocks}
    for graph_key, graph in _unique_graphs(samples).items():
        for block, names in _block_schemas((graph,)).items():
            if names and _rows_for_block(graph, block, names) and block not in state.blocks:
                raise GraphTransformError(
                    f"graph tensor block {block} was absent from the fitted training state"
                )
        data = graph.data.clone()
        for block, transform_state in state.blocks.items():
            rows = _rows_for_block(graph, block, transform_state.feature_order)
            if not rows:
                continue
            transformed = apply_feature_transform(rows, transform_state)
            _set_block_values(
                data, graph.metadata, block, transform_state.feature_order, transformed
            )
            rows_by_block[block].extend(rows)
        replacements[graph_key] = TensorizedGraph(data=data, metadata=dict(graph.metadata))
    return (
        tuple(replace(sample, graph=replacements[id(sample.graph)]) for sample in samples),
        rows_by_block,
    )


def _block_schemas(graphs: Sequence[TensorizedGraph]) -> dict[str, tuple[str, ...]]:
    schemas: dict[str, tuple[str, ...]] = {}
    for graph in graphs:
        feature_schema = graph.metadata.get("feature_schema")
        edge_schema = graph.metadata.get("edge_attr_schema")
        if not isinstance(feature_schema, Mapping) or not isinstance(edge_schema, Mapping):
            raise GraphTransformError("tensorized graph metadata is missing feature schemas")
        for node_type, raw_names in feature_schema.items():
            _set_schema(schemas, f"node::{node_type}", _eligible_names(raw_names))
        for edge_type, raw_names in edge_schema.items():
            _set_schema(schemas, f"edge::{edge_type}", _eligible_names(raw_names))
    return schemas


def _set_schema(schemas: dict[str, tuple[str, ...]], block: str, names: tuple[str, ...]) -> None:
    existing = schemas.get(block)
    if existing is not None and existing != names:
        raise GraphTransformError(f"inconsistent feature schema for graph tensor block {block}")
    schemas[block] = names


def _eligible_names(raw_names: object) -> tuple[str, ...]:
    if not isinstance(raw_names, list):
        raise GraphTransformError("tensorized graph feature schema must be a list")
    names = tuple(str(name) for name in raw_names)
    return tuple(
        name
        for name in names
        if infer_feature_kind(name)
        in {FeatureKind.CONTINUOUS, FeatureKind.BOUNDED_RATIO, FeatureKind.BINARY}
    )


def _rows_for_block(
    graph: TensorizedGraph, block: str, names: Sequence[str]
) -> list[dict[str, float]]:
    if not names:
        return []
    matrix, all_names = _block_matrix(graph, block)
    indices = _column_indices(all_names, names)
    return [
        {name: float(row[index]) for name, index in zip(names, indices, strict=True)}
        for row in matrix.detach().cpu().tolist()
    ]


def _set_block_values(
    data: Any,
    metadata: Mapping[str, object],
    block: str,
    names: Sequence[str],
    values: Sequence[Sequence[float]],
) -> None:
    matrix, all_names = _block_matrix_from_data(data, block, metadata)
    if matrix.size(0) != len(values):
        raise GraphTransformError(f"row count changed while applying graph block {block}")
    indices = _column_indices(all_names, names)
    updated = matrix.clone()
    for row_index, row in enumerate(values):
        for column_index, value in zip(indices, row, strict=True):
            updated[row_index, column_index] = float(value)
    _, target = _block_target(data, block)
    target.x = updated if block.startswith("node::") else getattr(target, "x", updated)
    if block.startswith("edge::"):
        target.edge_attr = updated


def _block_matrix(graph: TensorizedGraph, block: str) -> tuple[Any, tuple[str, ...]]:
    return _block_matrix_from_data(graph.data, block, graph.metadata)


def _block_matrix_from_data(
    data: Any, block: str, metadata: Mapping[str, object] | None = None
) -> tuple[Any, tuple[str, ...]]:
    metadata = metadata or getattr(data, "_seatflow_metadata", None)
    if metadata is None:
        raise GraphTransformError("graph metadata is required to resolve tensor feature schema")
    if block.startswith("node::"):
        node_type = block.removeprefix("node::")
        schemas = metadata.get("feature_schema")
        if not isinstance(schemas, Mapping) or not isinstance(schemas.get(node_type), list):
            raise GraphTransformError(f"missing node feature schema for {node_type}")
        return data[node_type].x, tuple(
            str(name) for name in cast(list[object], schemas[node_type])
        )
    if block.startswith("edge::"):
        edge_key = block.removeprefix("edge::")
        edge_type = tuple(edge_key.split("::"))
        schemas = metadata.get("edge_attr_schema")
        if not isinstance(schemas, Mapping) or not isinstance(schemas.get(edge_key), list):
            raise GraphTransformError(f"missing edge attribute schema for {edge_key}")
        return data[edge_type].edge_attr, tuple(
            str(name) for name in cast(list[object], schemas[edge_key])
        )
    raise GraphTransformError(f"unknown graph tensor block {block!r}")


def _block_target(data: Any, block: str) -> tuple[str, Any]:
    if block.startswith("node::"):
        return "node", data[block.removeprefix("node::")]
    if block.startswith("edge::"):
        return "edge", data[tuple(block.removeprefix("edge::").split("::"))]
    raise GraphTransformError(f"unknown graph tensor block {block!r}")


def _column_indices(all_names: Sequence[str], names: Sequence[str]) -> tuple[int, ...]:
    try:
        return tuple(all_names.index(name) for name in names)
    except ValueError as error:
        raise GraphTransformError(
            "fitted graph transform schema does not match tensor schema"
        ) from error


def _unique_graphs(samples: Sequence[TensorizedWindowSample]) -> dict[int, TensorizedGraph]:
    return {id(sample.graph): sample.graph for sample in samples}


def _graph_identity(graph: TensorizedGraph) -> str:
    return _hash_json(
        {
            "as_of_date": graph.metadata.get("as_of_date"),
            "mapping": graph.metadata.get("node_mapping"),
        }
    )


def _graph_sort_key(graph: TensorizedGraph) -> str:
    return _graph_identity(graph)


def _partition_diagnostics(
    rows_by_block: Mapping[str, Sequence[Mapping[str, float]]], state: GraphFeatureTransformState
) -> dict[str, object]:
    return {
        block: feature_transform_diagnostics(rows, state.blocks[block])
        for block, rows in rows_by_block.items()
        if rows
    }


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise GraphTransformError(f"{name} must be a SHA-256 hex digest")
