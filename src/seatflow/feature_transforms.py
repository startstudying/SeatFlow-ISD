"""Fold-local, serializable numeric feature transforms.

This module deliberately has no torch or PyG dependency.  A caller fits a state
from *only* the training partition, persists it next to a checkpoint, and applies
the exact state to validation/test partitions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import copysign, isfinite, log1p, sqrt
from typing import cast

from seatflow.feature_registry import FeatureKind, infer_feature_kind

TRANSFORM_STATE_VERSION = "seatflow_fold_transform_v1"
_AMOUNT_TOKENS = ("amount", "turnover", "money", "cash", "notional", "volume")
_WINSORIZED_TOKENS = ("return", "momentum", "volatility", "amplitude", "ratio")


class FeatureTransformError(ValueError):
    """Raised when a transform state or input cannot be safely used."""


@dataclass(frozen=True)
class FeatureStatistics:
    transform: str
    lower: float
    upper: float
    mean: float
    scale: float
    zero_variance_value: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform,
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "scale": self.scale,
            "zero_variance_value": self.zero_variance_value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> FeatureStatistics:
        try:
            zero_value = raw.get("zero_variance_value")
            return cls(
                transform=str(raw["transform"]),
                lower=float(cast(float, raw["lower"])),
                upper=float(cast(float, raw["upper"])),
                mean=float(cast(float, raw["mean"])),
                scale=float(cast(float, raw["scale"])),
                zero_variance_value=None if zero_value is None else float(cast(float, zero_value)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FeatureTransformError("invalid feature transform statistics") from error


@dataclass(frozen=True)
class FeatureTransformState:
    feature_order: tuple[str, ...]
    statistics: dict[str, FeatureStatistics]
    fitted_sample_count: int
    fitted_sample_ids_hash: str
    fitted_date_range: tuple[str, str] | None
    manifest_hash: str
    version: str = TRANSFORM_STATE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_order": list(self.feature_order),
            "statistics": {key: value.to_dict() for key, value in sorted(self.statistics.items())},
            "fitted_sample_count": self.fitted_sample_count,
            "fitted_sample_ids_hash": self.fitted_sample_ids_hash,
            "fitted_date_range": list(self.fitted_date_range) if self.fitted_date_range else None,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> FeatureTransformState:
        try:
            raw_order = raw["feature_order"]
            if not isinstance(raw_order, list):
                raise TypeError("feature_order must be a list")
            order = tuple(str(item) for item in raw_order)
            stats_raw = raw["statistics"]
            if not isinstance(stats_raw, Mapping):
                raise TypeError("statistics must be a mapping")
            statistics = {
                str(key): FeatureStatistics.from_dict(value)
                for key, value in stats_raw.items()
                if isinstance(value, Mapping)
            }
            state = cls(
                feature_order=order,
                statistics=statistics,
                fitted_sample_count=int(cast(int, raw["fitted_sample_count"])),
                fitted_sample_ids_hash=str(raw["fitted_sample_ids_hash"]),
                fitted_date_range=_date_range(raw.get("fitted_date_range")),
                manifest_hash=str(raw["manifest_hash"]),
                version=str(raw["version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FeatureTransformError("invalid feature transform state") from error
        state.validate()
        return state

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(payload).hexdigest()

    def validate(self) -> None:
        if self.version != TRANSFORM_STATE_VERSION:
            raise FeatureTransformError(f"unsupported transform state version: {self.version!r}")
        _validate_feature_order(self.feature_order)
        if set(self.statistics) != set(self.feature_order):
            raise FeatureTransformError("transform statistics do not match feature order")
        if self.fitted_sample_count < 1 or len(self.fitted_sample_ids_hash) != 64:
            raise FeatureTransformError(
                "transform state is not fitted to identifiable training samples"
            )
        if len(self.manifest_hash) != 64:
            raise FeatureTransformError("transform state is missing a valid manifest hash")
        for name, stats in self.statistics.items():
            if stats.transform not in {
                "signed_log_standardize",
                "robust_standardize",
                "winsorize_standardize",
                "identity",
            }:
                raise FeatureTransformError(f"unsupported transform for {name!r}")
            if not all(
                isfinite(value) for value in (stats.lower, stats.upper, stats.mean, stats.scale)
            ):
                raise FeatureTransformError(f"non-finite transform statistics for {name!r}")
            if stats.lower > stats.upper or stats.scale < 0:
                raise FeatureTransformError(f"invalid clipping or scale for {name!r}")


def fit_feature_transform(
    rows: Sequence[Mapping[str, float]],
    feature_order: Sequence[str],
    *,
    sample_ids: Sequence[str],
    manifest_hash: str,
    dates: Sequence[str] = (),
    winsor_quantile: float = 0.01,
    pretransformed_features: Sequence[str] = (),
) -> FeatureTransformState:
    """Fit a numeric transform from one fold's training rows only."""
    order = tuple(feature_order)
    _validate_feature_order(order)
    if not rows or len(rows) != len(sample_ids):
        raise FeatureTransformError("training rows and sample IDs must be non-empty and aligned")
    if not 0.0 <= winsor_quantile < 0.5:
        raise FeatureTransformError("winsor_quantile must be in [0, 0.5)")
    _validate_hash(manifest_hash, "manifest hash")
    if dates and len(dates) != len(rows):
        raise FeatureTransformError("dates must be empty or aligned with training rows")
    pretransformed = frozenset(pretransformed_features)
    if not pretransformed.issubset(order):
        raise FeatureTransformError("pretransformed features must belong to feature_order")
    values_by_feature: dict[str, list[float]] = {name: [] for name in order}
    for row in rows:
        _validate_row(row, order)
        for name in order:
            values_by_feature[name].append(float(row[name]))
    statistics = {
        name: _fit_statistics(
            name,
            values,
            winsor_quantile,
            forced_transform="robust_standardize" if name in pretransformed else None,
        )
        for name, values in values_by_feature.items()
    }
    state = FeatureTransformState(
        feature_order=order,
        statistics=statistics,
        fitted_sample_count=len(rows),
        fitted_sample_ids_hash=_string_sequence_hash(sample_ids),
        fitted_date_range=(min(dates), max(dates)) if dates else None,
        manifest_hash=manifest_hash,
    )
    state.validate()
    return state


def apply_feature_transform(
    rows: Sequence[Mapping[str, float]],
    state: FeatureTransformState,
    *,
    feature_order: Sequence[str] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Apply a persisted state; any schema or numeric violation fails closed."""
    state.validate()
    actual_order = tuple(feature_order) if feature_order is not None else state.feature_order
    if actual_order != state.feature_order:
        raise FeatureTransformError("feature order does not match fitted transform state")
    transformed: list[tuple[float, ...]] = []
    for row in rows:
        _validate_row(row, state.feature_order)
        output = tuple(
            _apply_value(float(row[name]), state.statistics[name]) for name in state.feature_order
        )
        if not all(isfinite(value) for value in output):
            raise FeatureTransformError("transform produced non-finite output")
        transformed.append(output)
    return tuple(transformed)


def feature_transform_diagnostics(
    rows: Sequence[Mapping[str, float]], state: FeatureTransformState
) -> dict[str, object]:
    """Return deterministic per-feature post-transform distribution evidence."""
    transformed = apply_feature_transform(rows, state)
    columns = zip(*transformed, strict=True) if transformed else ()
    distributions: dict[str, dict[str, float | int]] = {}
    for name, column in zip(state.feature_order, columns, strict=True):
        values = tuple(column)
        distributions[name] = {
            "count": len(values),
            "missing_count": 0,
            "missing_ratio": 0.0,
            "zero_count": sum(value == 0.0 for value in values),
            "zero_ratio": sum(value == 0.0 for value in values) / len(values),
            "min": min(values),
            "p01": _quantile(values, 0.01),
            "p05": _quantile(values, 0.05),
            "p50": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": max(values),
            "mean": sum(values) / len(values),
            "std": _population_std(values),
            "finite_count": sum(isfinite(value) for value in values),
        }
    result: dict[str, object] = {
        "transform_state_hash": state.fingerprint(),
        "feature_order": list(state.feature_order),
        "distributions": distributions,
    }
    result["diagnostics_hash"] = sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _fit_statistics(
    name: str,
    raw_values: Sequence[float],
    quantile: float,
    *,
    forced_transform: str | None = None,
) -> FeatureStatistics:
    transform = forced_transform or _transform_for(name)
    values = (
        [_signed_log1p(value) for value in raw_values]
        if transform == "signed_log_standardize"
        else list(raw_values)
    )
    if transform == "identity":
        return FeatureStatistics(transform, min(values), max(values), 0.0, 1.0)
    lower, upper = _quantile(values, quantile), _quantile(values, 1.0 - quantile)
    clipped = [min(max(value, lower), upper) for value in values]
    if transform == "robust_standardize":
        mean = _quantile(clipped, 0.5)
        scale = _quantile(clipped, 0.75) - _quantile(clipped, 0.25)
    else:
        mean, scale = sum(clipped) / len(clipped), _population_std(clipped)
    return FeatureStatistics(
        transform, lower, upper, mean, scale, clipped[0] if scale == 0.0 else None
    )


def _apply_value(raw_value: float, stats: FeatureStatistics) -> float:
    if stats.transform == "identity":
        return raw_value
    value = _signed_log1p(raw_value) if stats.transform == "signed_log_standardize" else raw_value
    if stats.zero_variance_value is not None:
        # A constant training feature has no learnable scale.  Mapping every
        # future value to zero preserves a finite, fold-local representation
        # without letting a test observation change the fitted state.
        return 0.0
    standardized = (min(max(value, stats.lower), stats.upper) - stats.mean) / stats.scale
    return min(max(standardized, -8.0), 8.0)


def _transform_for(name: str) -> str:
    kind = infer_feature_kind(name)
    if kind in {
        FeatureKind.ENTITY_ID,
        FeatureKind.CATEGORICAL,
        FeatureKind.TIMESTAMP,
        FeatureKind.MASK,
    }:
        raise FeatureTransformError(
            f"non-continuous feature {name!r} cannot enter numeric transform"
        )
    normalized = name.lower()
    if kind in {FeatureKind.BINARY, FeatureKind.BOUNDED_RATIO}:
        return "identity"
    if any(token in normalized for token in _AMOUNT_TOKENS):
        return "signed_log_standardize"
    if any(token in normalized for token in _WINSORIZED_TOKENS):
        return "winsorize_standardize"
    return "winsorize_standardize"


def _validate_feature_order(order: Sequence[str]) -> None:
    if not order or len(set(order)) != len(order):
        raise FeatureTransformError("feature order must be non-empty and unique")
    for name in order:
        _transform_for(name)


def _validate_row(row: Mapping[str, float], order: Sequence[str]) -> None:
    if set(row) != set(order):
        raise FeatureTransformError("row features do not exactly match transform schema")
    for name in order:
        try:
            value = float(row[name])
        except (TypeError, ValueError) as error:
            raise FeatureTransformError(f"feature {name!r} is not numeric") from error
        if not isfinite(value):
            raise FeatureTransformError(f"feature {name!r} is non-finite")


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), int(position + 0.999999999)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _signed_log1p(value: float) -> float:
    return copysign(log1p(abs(value)), value)


def _string_sequence_hash(values: Sequence[str]) -> str:
    return sha256(json.dumps(list(values), separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise FeatureTransformError(f"{name} must be a SHA-256 hex digest")


def _date_range(raw: object) -> tuple[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2 or not all(isinstance(item, str) for item in raw):
        raise FeatureTransformError("invalid fitted date range")
    return raw[0], raw[1]
