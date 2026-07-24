"""Typed return-unit conversion and deterministic aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import cast


class ReturnUnit(StrEnum):
    """Canonical units accepted by return payloads."""

    DECIMAL = "decimal"
    PERCENT = "percent"
    BPS = "bps"


class ReturnAggregationMethod(StrEnum):
    """Supported reduction rules for multi-day return sequences."""

    DAILY_ARITHMETIC_MEAN = "daily_arithmetic_mean"
    SUM = "sum"
    COMPOUND = "compound"


def _as_float(value: object) -> float:
    return float(cast(str | int | float | bool, value))


def _to_decimal(values: tuple[float, ...], unit: ReturnUnit) -> tuple[float, ...]:
    if unit == ReturnUnit.DECIMAL:
        return values
    if unit == ReturnUnit.PERCENT:
        return tuple(value / 100.0 for value in values)
    if unit == ReturnUnit.BPS:
        return tuple(value / 10_000.0 for value in values)
    raise ValueError(f"unsupported return unit: {unit}")


def _aggregate(values: tuple[float, ...], method: ReturnAggregationMethod) -> float:
    if method == ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN:
        return fmean(values)
    if method == ReturnAggregationMethod.SUM:
        return sum(values)
    if method == ReturnAggregationMethod.COMPOUND:
        product = 1.0
        for value in values:
            product *= 1.0 + value
        return product - 1.0
    raise ValueError(f"unsupported aggregation method: {method}")


@dataclass(frozen=True)
class ReturnUnits:
    """Canonical multi-unit return payload."""

    decimal: float
    percent: float
    bps: float
    aggregation_method: ReturnAggregationMethod
    number_of_days: int

    def to_dict(self) -> dict[str, object]:
        return {
            "decimal": self.decimal,
            "percent": self.percent,
            "bps": self.bps,
            "aggregation_method": self.aggregation_method.value,
            "number_of_days": self.number_of_days,
        }


def return_units(
    values: tuple[float, ...] | list[float],
    *,
    unit: ReturnUnit = ReturnUnit.DECIMAL,
    aggregation_method: ReturnAggregationMethod = ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
    number_of_days: int | None = None,
) -> ReturnUnits:
    """Build strict-return payload with all unit projections and aggregation metadata."""

    if not values:
        raise ValueError("return_units requires at least one value")
    if number_of_days is not None and number_of_days <= 0:
        raise ValueError("number_of_days must be positive")
    numeric = tuple(_as_float(value) for value in values)
    decimal_values = _to_decimal(numeric, unit)
    method = aggregation_method
    aggregated = _aggregate(decimal_values, method)
    days = len(decimal_values) if number_of_days is None else number_of_days
    return ReturnUnits(
        decimal=aggregated,
        percent=aggregated * 100.0,
        bps=aggregated * 10_000.0,
        aggregation_method=method,
        number_of_days=days,
    )


def summarize_daily_return(
    value: float,
    *,
    unit: ReturnUnit = ReturnUnit.DECIMAL,
) -> ReturnUnits:
    """Single-day convenience wrapper that still emits full return-unit metadata."""

    return return_units(
        (value,),
        unit=unit,
        aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
    )
