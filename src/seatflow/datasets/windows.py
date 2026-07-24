from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Protocol

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.types import HeteroGraphSnapshot
from seatflow.labels import (
    FIXTURE_ADJUSTMENT_POLICY,
    FIXTURE_CALENDAR_POLICY,
    FROZEN_ADJUSTMENT_POLICY,
    FROZEN_CALENDAR_POLICY,
    LABEL_V2_ADJUSTMENT_POLICY,
    RESEARCH_HORIZONS,
    StockLabel,
    StockLabelV2,
    generate_future_return_labels,
    label_distribution,
)

CutoffResolver = Callable[[date], datetime]
WindowLabel = StockLabel | StockLabelV2


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


@dataclass(frozen=True)
class WindowSample:
    sample_id: str
    as_of_date: date
    cutoff_time: datetime
    target_stock_id: str
    graph: HeteroGraphSnapshot
    labels: tuple[WindowLabel, ...]
    source_hash: str

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("window sample requires labels")
        if len({type(label) for label in self.labels}) != 1:
            raise ValueError("window sample must not mix legacy and V2 labels")
        if any(label.target_stock_id != self.target_stock_id for label in self.labels):
            raise ValueError("window sample label target does not match sample target")
        if any(label.as_of_date != self.as_of_date for label in self.labels):
            raise ValueError("window sample label as-of date does not match sample")
        if any(label.cutoff_time != self.cutoff_time for label in self.labels):
            raise ValueError("window sample label cutoff does not match sample")

    def manifest(self) -> dict[str, object]:
        label_windows = [_label_manifest(label) for label in self.labels]
        return {
            "sample_id": self.sample_id,
            "as_of_date": self.as_of_date.isoformat(),
            "cutoff_time": self.cutoff_time.isoformat(),
            "target_stock_id": self.target_stock_id,
            "label_window": label_windows,
            "label_windows": label_windows,
            "source_hash": self.source_hash,
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
        }


@dataclass(frozen=True)
class DatasetManifest:
    samples: tuple[WindowSample, ...]
    split: Mapping[str, Sequence[str]]
    tensor_shapes: Mapping[str, object]
    missingness: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        labels = [label for sample in self.samples for label in sample.labels]
        sample_records = [sample.manifest() for sample in self.samples]
        readiness = _data_readiness(self.samples, self.missingness)
        return {
            "sample_count": len(self.samples),
            "samples": sample_records,
            "split": {key: list(value) for key, value in self.split.items()},
            "shape": dict(self.tensor_shapes),
            "feature_schema": _feature_schema(),
            "label_schema": _label_schema(),
            "label_distribution": _label_distribution(labels),
            "horizon_distribution": _horizon_distribution(labels),
            "missingness": dict(self.missingness),
            "missing_reasons": dict(self.missingness),
            "source_hash": _manifest_hash(sample_records, self.split, self.tensor_shapes),
            "data_readiness": readiness,
            "claims": {
                "engineering_complete": bool(readiness["real_research_ready"]),
                "data_ready": bool(readiness["real_research_ready"]),
                "model_valid": False,
            },
        }


def build_window_samples(
    records: Iterable[DragonTigerRecord],
    bars: Iterable[DailyBar],
    as_of_dates: Sequence[date],
    target_stock_ids: Sequence[str],
    horizons: Sequence[int],
    cutoff_for_date: CutoffResolver,
    decay_lambda: float,
    max_edge_age_days: int,
    adjustment_policy: str = FIXTURE_ADJUSTMENT_POLICY,
    calendar_policy: str = FIXTURE_CALENDAR_POLICY,
    trading_calendar: Sequence[date] | None = None,
    matured_labels: Iterable[StockLabel] = (),
    v2_labels_by_key: Mapping[tuple[date, str], Sequence[StockLabelV2]] | None = None,
    target_stock_ids_by_date: Mapping[date, Sequence[str]] | None = None,
    correlation_stock_ids_by_date: Mapping[date, Sequence[str]] | None = None,
    graph_stock_ids_by_date: Mapping[date, Sequence[str]] | None = None,
    cooccurrence_max_records_per_stock: int | None = None,
) -> tuple[WindowSample, ...]:
    builder = DynamicGraphBuilder(
        decay_lambda,
        max_edge_age_days,
        cooccurrence_max_records_per_stock=cooccurrence_max_records_per_stock,
    )
    all_records = tuple(records)
    all_bars = tuple(bars)
    all_matured_labels = tuple(matured_labels)
    records_by_date = _records_by_date(all_records)
    bars_by_stock = _bars_by_stock(all_bars)
    samples: list[WindowSample] = []
    for as_of_date in sorted(as_of_dates):
        cutoff_time = cutoff_for_date(as_of_date)
        date_target_stock_ids = (
            target_stock_ids_by_date.get(as_of_date, ())
            if target_stock_ids_by_date is not None
            else target_stock_ids
        )
        graph_records = _records_in_graph_window(
            records_by_date, as_of_date, max_edge_age_days
        )
        if graph_stock_ids_by_date is not None:
            graph_stock_ids = set(graph_stock_ids_by_date.get(as_of_date, ()))
            graph_records = tuple(
                record for record in graph_records if record.stock_id in graph_stock_ids
            )
        correlation_stock_ids = (
            correlation_stock_ids_by_date.get(as_of_date, tuple(bars_by_stock))
            if correlation_stock_ids_by_date is not None
            else tuple(bars_by_stock)
        )
        graph = builder.build(
            graph_records,
            as_of_date,
            cutoff_time=cutoff_time,
            matured_labels=tuple(
                label for label in all_matured_labels if label.as_of_date <= as_of_date
            ),
            profile_records=_records_through_date(records_by_date, as_of_date),
            bars=_recent_bars_for_stocks(bars_by_stock, correlation_stock_ids, as_of_date),
        )
        graph_hash = _graph_hash(graph)
        for target_stock_id in sorted(set(date_target_stock_ids)):
            if v2_labels_by_key is not None:
                v2_labels = tuple(
                    v2_labels_by_key.get((as_of_date, target_stock_id), ())
                )
                if v2_labels and (
                    any(label.disposition != "eligible" for label in v2_labels)
                    or tuple(sorted(label.horizon for label in v2_labels))
                    != tuple(sorted(set(horizons)))
                ):
                    continue
                labels: tuple[WindowLabel, ...] = v2_labels
            else:
                target_bars = bars_by_stock.get(target_stock_id)
                if target_bars is None:
                    continue
                labels = generate_future_return_labels(
                    target_bars,
                    as_of_date=as_of_date,
                    cutoff_time=cutoff_time,
                    target_stock_ids=(target_stock_id,),
                    horizons=horizons,
                    adjustment_policy=adjustment_policy,
                    calendar_policy=calendar_policy,
                    trading_calendar=trading_calendar,
                )
            if not labels:
                continue
            sample_id = f"{as_of_date.isoformat()}::{target_stock_id}"
            samples.append(
                WindowSample(
                    sample_id=sample_id,
                    as_of_date=as_of_date,
                    cutoff_time=cutoff_time,
                    target_stock_id=target_stock_id,
                    graph=graph,
                    labels=labels,
                    source_hash=_sample_hash(graph_hash, labels),
                )
            )
    return tuple(samples)


def write_dataset_manifest(path: str, manifest: DatasetManifest) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)


def _sample_hash(graph_hash: str, labels: tuple[WindowLabel, ...]) -> str:
    payload = {"graph_hash": graph_hash, "labels": [label.to_dict() for label in labels]}
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _graph_hash(graph: HeteroGraphSnapshot) -> str:
    digest = sha256()
    _update_graph_hash(digest, ("as_of_date", graph.as_of_date.isoformat()))
    for node in sorted(graph.nodes, key=lambda item: (item.node_type.value, item.node_id)):
        _update_graph_hash(digest, ("node", node.node_type.value, node.node_id, node.features))
    for edge in sorted(
        graph.edges,
        key=lambda item: (
            item.event_date,
            item.source_type.value,
            item.source_id,
            item.relation,
            item.target_type.value,
            item.target_id,
        ),
    ):
        _update_graph_hash(
            digest,
            (
                "edge",
                edge.source_type.value,
                edge.source_id,
                edge.relation,
                edge.target_type.value,
                edge.target_id,
                edge.weight,
                edge.event_date.isoformat(),
                edge.attributes,
            ),
        )
    return digest.hexdigest()


def _update_graph_hash(digest: _Digest, payload: object) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, default=str, separators=(",", ":")
    ).encode("utf-8")
    digest.update(encoded)
    digest.update(b"\n")


def _feature_schema() -> dict[str, object]:
    return {
        "graph": {
            "nodes": ["node_type", "node_id", "features"],
            "edges": [
                "source_type",
                "source_id",
                "relation",
                "target_type",
                "target_id",
                "weight",
                "event_date",
                "attributes",
            ],
            "temporal_contract": "features must be available at or before cutoff_time",
        }
    }


def _label_schema() -> dict[str, object]:
    return {
        "target": "stock",
        "fields": [
            "horizon",
            "return_target",
            "direction_label",
            "label_start_date",
            "label_end_date",
            "label_available_time",
            "adjustment_policy",
            "calendar_policy",
            "base_source_name",
            "future_source_name",
            "base_source_version",
            "future_source_version",
        ],
        "window_contract": "label_start_date is t+1 trading day after sample as_of_date",
    }


def _horizon_distribution(labels: Sequence[WindowLabel]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for label in labels:
        key = str(label.horizon)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _data_readiness(
    samples: Sequence[WindowSample],
    missingness: Mapping[str, int],
) -> dict[str, object]:
    labels = [label for sample in samples for label in sample.labels]
    frozen_policy_ready = _frozen_one_three_five_policy(labels)
    source_hashes_verified = _source_hashes_verified(samples)
    no_missingness = _no_reported_missingness(missingness)
    real_research_ready = (
        bool(samples) and frozen_policy_ready and source_hashes_verified and no_missingness
    )
    blocked_reasons: list[str] = []
    if not samples:
        blocked_reasons.append("no_samples")
    if not frozen_policy_ready:
        blocked_reasons.append("non_frozen_or_incomplete_1_3_5_labeling_policy")
    if not source_hashes_verified:
        blocked_reasons.append("invalid_sample_source_hash")
    if not no_missingness:
        blocked_reasons.append("reported_missingness_nonzero")
    return {
        "real_research_ready": real_research_ready,
        "frozen_1_3_5_policy": frozen_policy_ready,
        "source_hashes_verified": source_hashes_verified,
        "no_reported_missingness": no_missingness,
        "reason": "ready" if real_research_ready else ";".join(blocked_reasons),
    }


def _records_by_date(
    records: Sequence[DragonTigerRecord],
) -> dict[date, tuple[DragonTigerRecord, ...]]:
    grouped: dict[date, list[DragonTigerRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trade_date].append(record)
    return {trade_date: tuple(values) for trade_date, values in grouped.items()}


def _records_in_graph_window(
    records_by_date: Mapping[date, Sequence[DragonTigerRecord]],
    as_of_date: date,
    max_edge_age_days: int,
) -> tuple[DragonTigerRecord, ...]:
    return tuple(
        record
        for trade_date, records in records_by_date.items()
        if 0 <= (as_of_date - trade_date).days <= max_edge_age_days
        for record in records
    )


def _records_through_date(
    records_by_date: Mapping[date, Sequence[DragonTigerRecord]],
    as_of_date: date,
) -> tuple[DragonTigerRecord, ...]:
    return tuple(
        record
        for trade_date, records in records_by_date.items()
        if trade_date <= as_of_date
        for record in records
    )


def _bars_by_stock(bars: Sequence[DailyBar]) -> dict[str, tuple[DailyBar, ...]]:
    grouped: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.stock_id].append(bar)
    return {
        stock_id: tuple(sorted(values, key=lambda item: item.trade_date))
        for stock_id, values in grouped.items()
    }


def _recent_bars_for_stocks(
    bars_by_stock: Mapping[str, Sequence[DailyBar]],
    stock_ids: Sequence[str],
    as_of_date: date,
    *,
    lookback_days: int = 61,
) -> tuple[DailyBar, ...]:
    """Keep correlation inputs bounded to the requested as-of stock universe."""

    selected: list[DailyBar] = []
    for stock_id in sorted(set(stock_ids)):
        stock_bars = bars_by_stock.get(stock_id, ())
        dates = [bar.trade_date for bar in stock_bars]
        end = bisect_right(dates, as_of_date)
        selected.extend(stock_bars[max(0, end - lookback_days) : end])
    return tuple(selected)


def _frozen_one_three_five_policy(labels: Sequence[WindowLabel]) -> bool:
    if not labels:
        return False
    return (
        tuple(sorted({label.horizon for label in labels})) == RESEARCH_HORIZONS
        and all(
            label.adjustment_policy
            in {FROZEN_ADJUSTMENT_POLICY, LABEL_V2_ADJUSTMENT_POLICY}
            for label in labels
        )
        and all(label.calendar_policy == FROZEN_CALENDAR_POLICY for label in labels)
        and len({type(label) for label in labels}) == 1
    )


def _label_return(label: WindowLabel) -> object:
    return label.future_return if isinstance(label, StockLabel) else label.excess_return


def _label_schema_version(label: WindowLabel) -> str:
    if isinstance(label, StockLabel):
        return "future_return_binary_v1"
    return "executable_excess_return_v2"


def _label_manifest(label: WindowLabel) -> dict[str, object]:
    payload: dict[str, object] = {
        "horizon": label.horizon,
        "start": label.label_start_date.isoformat(),
        "end": label.label_end_date.isoformat(),
        "available_time": label.label_available_time.isoformat(),
        "return_target": str(_label_return(label)),
        "direction_label": label.direction_label,
        "label_schema_version": _label_schema_version(label),
        "disposition": getattr(label, "disposition", "eligible"),
    }
    if isinstance(label, StockLabel):
        payload["future_return"] = str(label.future_return)
    else:
        payload["excess_return"] = str(label.excess_return)
        payload["benchmark_id"] = label.benchmark_id
        payload["source_hash"] = label.source_hash
    return payload


def _label_distribution(labels: Sequence[WindowLabel]) -> dict[str, int]:
    if all(isinstance(label, StockLabel) for label in labels):
        return label_distribution(label for label in labels if isinstance(label, StockLabel))
    counts: dict[str, int] = {}
    for label in labels:
        if label.direction_label is None:
            raise ValueError("dataset manifest cannot include an ineligible V2 label")
        key = str(label.direction_label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_hashes_verified(samples: Sequence[WindowSample]) -> bool:
    return bool(samples) and all(_is_sha256_hex(sample.source_hash) for sample in samples)


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _no_reported_missingness(missingness: Mapping[str, int]) -> bool:
    return all(count == 0 for count in missingness.values())


def _manifest_hash(
    samples: Sequence[Mapping[str, object]],
    split: Mapping[str, Sequence[str]],
    tensor_shapes: Mapping[str, object],
) -> str:
    payload = {
        "samples": samples,
        "split": {key: list(value) for key, value in split.items()},
        "tensor_shapes": dict(tensor_shapes),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
