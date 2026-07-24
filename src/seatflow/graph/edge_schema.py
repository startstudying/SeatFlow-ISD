"""Strongly typed validation and transforms for numeric edge attributes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, log1p

from seatflow.feature_registry import FeatureKind, infer_feature_kind


class EdgeFeatureType(StrEnum):
    CONTINUOUS = "continuous"
    RATIO = "ratio"
    COUNT = "count"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class EdgeField:
    name: str
    kind: EdgeFeatureType
    relative_time: bool = False


_RELATIVE_TIME_FIELDS = frozenset({"delta_days", "last_trade_age"})
_SIGNED_RATIO_FIELDS = frozenset(
    {"direction", "source_direction", "target_direction", "correlation", "net_buy_ratio"}
)


def classify_edge_field(name: str) -> EdgeField:
    normalized = name.lower()
    feature_kind = infer_feature_kind(normalized)
    if feature_kind == FeatureKind.ENTITY_ID:
        return EdgeField(name, EdgeFeatureType.IDENTIFIER)
    if normalized in _RELATIVE_TIME_FIELDS:
        return EdgeField(name, EdgeFeatureType.TIMESTAMP, relative_time=True)
    if feature_kind == FeatureKind.TIMESTAMP:
        return EdgeField(name, EdgeFeatureType.TIMESTAMP)
    if normalized.endswith("_count") or normalized in {"overlap_days", "trade_count"}:
        return EdgeField(name, EdgeFeatureType.COUNT)
    if (
        feature_kind in {FeatureKind.BOUNDED_RATIO, FeatureKind.BINARY}
        or "_share_" in normalized
        or normalized.startswith("amount_share_")
        or normalized in _SIGNED_RATIO_FIELDS
    ):
        return EdgeField(name, EdgeFeatureType.RATIO)
    if feature_kind == FeatureKind.CATEGORICAL:
        return EdgeField(name, EdgeFeatureType.CATEGORICAL)
    return EdgeField(name, EdgeFeatureType.CONTINUOUS)


def validate_edge_attributes(attributes: Mapping[str, int | float]) -> None:
    for name, raw_value in attributes.items():
        field = classify_edge_field(name)
        value = float(raw_value)
        if not isfinite(value):
            raise ValueError(f"edge attribute {name!r} must be finite")
        if field.kind == EdgeFeatureType.IDENTIFIER:
            raise ValueError(
                f"edge attributes contain entity identifiers: {name!r} "
                "cannot enter the continuous MLP"
            )
        if field.kind == EdgeFeatureType.CATEGORICAL:
            raise ValueError(f"categorical edge field {name!r} requires an embedding or relation")
        if field.kind == EdgeFeatureType.TIMESTAMP:
            if not field.relative_time:
                raise ValueError(
                    f"timestamp edge field {name!r} must be converted to relative time"
                )
            if value < 0:
                raise ValueError(f"relative time edge field {name!r} must be non-negative")
        if field.kind == EdgeFeatureType.COUNT and (value < 0 or not value.is_integer()):
            raise ValueError(f"count edge field {name!r} must be a non-negative integer")
        if field.kind == EdgeFeatureType.RATIO:
            lower = -1.0 if name.lower() in _SIGNED_RATIO_FIELDS else 0.0
            if not lower <= value <= 1.0:
                raise ValueError(
                    f"ratio edge field {name!r} must be in [{lower}, 1.0]"
                )


def transform_edge_value(name: str, value: float) -> float:
    field = classify_edge_field(name)
    validate_edge_attributes({name: value})
    if field.kind == EdgeFeatureType.COUNT:
        return log1p(value)
    return value
