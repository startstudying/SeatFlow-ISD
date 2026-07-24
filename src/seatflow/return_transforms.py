"""Fold-local return-label scaling for the DT-SeatHGT return head."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt

from seatflow.datasets.pyg import TensorizedWindowSample

RETURN_TRANSFORM_STATE_VERSION = "seatflow_return_label_v1"


class ReturnTransformError(ValueError):
    """Raised when return-label scaling cannot be safely applied."""


@dataclass(frozen=True)
class ReturnHorizonStatistics:
    mean: float
    std: float
    count: int

    def to_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "std": self.std, "count": self.count}


@dataclass(frozen=True)
class ReturnTransformState:
    horizons: tuple[int, ...]
    statistics: dict[int, ReturnHorizonStatistics]
    fitted_sample_ids_hash: str
    training_manifest_hash: str
    entry_price_definition: str = "as_of_close"
    exit_price_definition: str = "future_close_at_horizon"
    version: str = RETURN_TRANSFORM_STATE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "label_schema_version": self.version,
            "horizons": list(self.horizons),
            "return_statistics": {
                str(horizon): statistics.to_dict()
                for horizon, statistics in sorted(self.statistics.items())
            },
            "fitted_sample_ids_hash": self.fitted_sample_ids_hash,
            "training_manifest_hash": self.training_manifest_hash,
            "entry_price_definition": self.entry_price_definition,
            "exit_price_definition": self.exit_price_definition,
        }

    def fingerprint(self) -> str:
        return _hash_json(self.to_dict())

    def validate(self) -> None:
        if self.version != RETURN_TRANSFORM_STATE_VERSION:
            raise ReturnTransformError(f"unsupported return-label schema: {self.version!r}")
        if not self.horizons or set(self.horizons) != set(self.statistics):
            raise ReturnTransformError("return state horizons do not match statistics")
        _validate_hash(self.fitted_sample_ids_hash, "fitted sample IDs hash")
        _validate_hash(self.training_manifest_hash, "training manifest hash")
        for horizon, statistics in self.statistics.items():
            if horizon <= 0 or statistics.count < 1:
                raise ReturnTransformError("return state has an invalid horizon or sample count")
            if not isfinite(statistics.mean) or not isfinite(statistics.std) or statistics.std <= 0:
                raise ReturnTransformError("return state has non-finite or zero standard deviation")

    def transform(self, horizon: int, value: float) -> float:
        statistics = self._statistics_for(horizon)
        _require_finite(value)
        return (value - statistics.mean) / statistics.std

    def inverse_transform(self, horizon: int, value: float) -> float:
        statistics = self._statistics_for(horizon)
        _require_finite(value)
        result = value * statistics.std + statistics.mean
        _require_finite(result)
        return result

    def _statistics_for(self, horizon: int) -> ReturnHorizonStatistics:
        self.validate()
        try:
            return self.statistics[horizon]
        except KeyError as error:
            raise ReturnTransformError(f"return state is missing horizon {horizon}") from error


def fit_return_transform(samples: Sequence[TensorizedWindowSample]) -> ReturnTransformState:
    """Fit mean/std from one training fold only."""
    if not samples:
        raise ReturnTransformError("return transform requires non-empty training samples")
    horizons = tuple(sorted(samples[0].returns))
    if not horizons:
        raise ReturnTransformError("training samples have no return horizons")
    values: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    for sample in samples:
        if tuple(sorted(sample.returns)) != horizons:
            raise ReturnTransformError("training samples do not share return horizons")
        for horizon in horizons:
            value = float(sample.returns[horizon])
            _require_finite(value)
            values[horizon].append(value)
    statistics = {
        horizon: _statistics(horizon_values) for horizon, horizon_values in values.items()
    }
    state = ReturnTransformState(
        horizons=horizons,
        statistics=statistics,
        fitted_sample_ids_hash=_hash_json([sample.sample_id for sample in samples]),
        training_manifest_hash=_hash_json(
            [
                {"sample_id": sample.sample_id, "source_hash": sample.source_hash}
                for sample in samples
            ]
        ),
    )
    state.validate()
    return state


def return_transform_diagnostics(
    samples: Sequence[TensorizedWindowSample], state: ReturnTransformState
) -> dict[str, object]:
    state.validate()
    diagnostics: dict[str, object] = {
        "return_transform_hash": state.fingerprint(),
        "horizons": {},
    }
    horizon_diagnostics = diagnostics["horizons"]
    assert isinstance(horizon_diagnostics, dict)
    for horizon in state.horizons:
        standardized = [
            state.transform(horizon, float(sample.returns[horizon])) for sample in samples
        ]
        horizon_diagnostics[str(horizon)] = {
            "count": len(standardized),
            "mean": sum(standardized) / len(standardized),
            "std": _std(standardized),
            "min": min(standardized),
            "max": max(standardized),
        }
    diagnostics["diagnostics_hash"] = _hash_json(diagnostics)
    return diagnostics


def _statistics(values: Sequence[float]) -> ReturnHorizonStatistics:
    mean = sum(values) / len(values)
    std = _std(values)
    # A zero-variance label is not a learnable return target and must not be silently masked.
    if std == 0.0:
        raise ReturnTransformError("return labels have zero variance in the training fold")
    return ReturnHorizonStatistics(mean=mean, std=std, count=len(values))


def _std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _require_finite(value: float) -> None:
    if not isfinite(value):
        raise ReturnTransformError("return value must be finite")


def _hash_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ReturnTransformError(f"{name} must be a SHA-256 hex digest")
