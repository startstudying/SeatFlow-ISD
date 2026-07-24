from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import cast

from seatflow.datasets.windows import WindowSample
from seatflow.graph.tensorize import (
    FrozenGraphSchema,
    TensorizedGraph,
    derive_frozen_graph_schema,
    tensorize_hetero_graph,
)
from seatflow.labels import (
    FROZEN_ADJUSTMENT_POLICY,
    FROZEN_CALENDAR_POLICY,
    LABEL_V2_ADJUSTMENT_POLICY,
    StockLabel,
    StockLabelV2,
)

LEGACY_BINARY_LABEL_SCHEMA = "future_return_binary_v1"
EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA = "executable_excess_return_v2"


@dataclass(frozen=True)
class TensorizedWindowSample:
    sample_id: str
    source_hash: str
    target_stock_id: str
    target_node_index: int
    graph: TensorizedGraph
    labels: dict[int, int]
    returns: dict[int, float]
    as_of_date: date | None = None
    label_end_dates: Mapping[int, date] = field(default_factory=dict)
    label_schema_version: str = LEGACY_BINARY_LABEL_SCHEMA

    def __post_init__(self) -> None:
        if self.label_schema_version not in {
            LEGACY_BINARY_LABEL_SCHEMA,
            EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        }:
            raise ValueError("unsupported tensorized label schema version")
        expected_labels = (
            {0, 1}
            if self.label_schema_version == LEGACY_BINARY_LABEL_SCHEMA
            else {-1, 0, 1}
        )
        if any(label not in expected_labels for label in self.labels.values()):
            raise ValueError("tensorized sample label does not match declared label schema")


def tensorize_window_sample(sample: WindowSample) -> TensorizedWindowSample:
    graph = tensorize_hetero_graph(sample.graph)
    return _tensorized_window_sample(sample, graph)


def tensorize_window_samples(
    samples: Iterable[WindowSample],
    *,
    frozen_schema: FrozenGraphSchema | None = None,
) -> tuple[TensorizedWindowSample, ...]:
    materialized = tuple(samples)
    if not materialized:
        return ()
    for sample in materialized:
        _require_frozen_labeling_policy(sample)
    unique_snapshots = tuple(
        {id(sample.graph): sample.graph for sample in materialized}.values()
    )
    schema = frozen_schema or derive_frozen_graph_schema(unique_snapshots)
    tensorized_by_graph: dict[int, TensorizedGraph] = {}
    tensorized_samples: list[TensorizedWindowSample] = []
    for sample in materialized:
        graph_key = id(sample.graph)
        graph = tensorized_by_graph.get(graph_key)
        if graph is None:
            graph = tensorize_hetero_graph(sample.graph, frozen_schema=schema)
            tensorized_by_graph[graph_key] = graph
        tensorized_samples.append(_tensorized_window_sample(sample, graph))
    return tuple(tensorized_samples)


def _tensorized_window_sample(
    sample: WindowSample,
    graph: TensorizedGraph,
) -> TensorizedWindowSample:
    node_mapping = graph.metadata.get("node_mapping")
    if not isinstance(node_mapping, dict):
        raise ValueError("tensorized graph metadata missing node mapping")
    stock_mapping = cast(dict[str, object], node_mapping).get("stock", {})
    if not isinstance(stock_mapping, dict):
        raise ValueError("tensorized graph metadata missing stock node mapping")
    if sample.target_stock_id not in stock_mapping:
        raise ValueError(
            f"target stock {sample.target_stock_id!r} is missing from tensorized graph"
        )
    return TensorizedWindowSample(
        sample_id=sample.sample_id,
        source_hash=sample.source_hash,
        target_stock_id=sample.target_stock_id,
        target_node_index=int(stock_mapping[sample.target_stock_id]),
        graph=graph,
        labels={label.horizon: _direction_label(label) for label in sample.labels},
        returns={label.horizon: _return_target(label) for label in sample.labels},
        as_of_date=sample.as_of_date,
        label_end_dates={label.horizon: label.label_end_date for label in sample.labels},
        label_schema_version=_label_schema_version(sample.labels),
    )


def _require_frozen_labeling_policy(sample: WindowSample) -> None:
    if not sample.labels:
        raise ValueError(f"sample {sample.sample_id} has no labels")
    label_types = {type(label) for label in sample.labels}
    if label_types == {StockLabel}:
        expected_adjustment_policy = FROZEN_ADJUSTMENT_POLICY
    elif label_types == {StockLabelV2}:
        expected_adjustment_policy = LABEL_V2_ADJUSTMENT_POLICY
        if any(
            not isinstance(label, StockLabelV2) or label.disposition != "eligible"
            for label in sample.labels
        ):
            raise ValueError("tensorized V2 samples require eligible labels")
    else:
        raise ValueError("tensorized window sample must not mix label schemas")
    if any(label.adjustment_policy != expected_adjustment_policy for label in sample.labels):
        raise ValueError("tensorized window samples require frozen adjustment policy")
    if any(label.calendar_policy != FROZEN_CALENDAR_POLICY for label in sample.labels):
        raise ValueError("tensorized window samples require frozen trading calendar policy")


def _direction_label(label: StockLabel | StockLabelV2) -> int:
    if label.direction_label is None:
        raise ValueError("tensorized V2 samples require direction labels")
    return label.direction_label


def _return_target(label: StockLabel | StockLabelV2) -> float:
    value = label.future_return if isinstance(label, StockLabel) else label.excess_return
    if value is None:
        raise ValueError("tensorized V2 samples require excess return")
    return float(value)


def _label_schema_version(labels: tuple[StockLabel | StockLabelV2, ...]) -> str:
    return (
        LEGACY_BINARY_LABEL_SCHEMA
        if all(isinstance(label, StockLabel) for label in labels)
        else EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA
    )
