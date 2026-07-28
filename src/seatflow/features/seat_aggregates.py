"""Seat aggregate features for E2 experiments.

This module computes stock-level aggregate features from DragonTiger seat data,
providing a configurable interface for E0-E4 experiments.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite, log1p
from typing import Any

from seatflow.data.contracts import normalize_stock_id
from seatflow.data.records import (
    INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    DragonTigerRecord,
    TradeSide,
)

# Seat aggregate feature names (aligned with r4_features/seat_aggregates.py)
SEAT_AGGREGATE_FEATURES: tuple[str, ...] = (
    "seat_buy_amount",
    "seat_sell_amount",
    "seat_net_amount",
    "seat_count",
    "seat_buy_count",
    "seat_sell_count",
    "institution_seat_ratio",
    "northbound_seat_ratio",
    "seat_concentration",
    "seat_repeat_participation",
    "seat_signal_strength",
)

# Missing event marker feature
MISSING_SEAT_EVENT_FEATURE = "missing_seat_event"

# Full feature set including missing marker
SEAT_AGGREGATE_MODEL_FEATURES: tuple[str, ...] = (
    *SEAT_AGGREGATE_FEATURES,
    MISSING_SEAT_EVENT_FEATURE,
)

# Feature groups by experiment level
SEAT_BASELINE: tuple[str, ...] = (
    "seat_buy_amount",
    "seat_sell_amount",
    "seat_net_amount",
    "seat_count",
)

SEAT_E0: tuple[str, ...] = (
    "seat_buy_amount",
    "seat_sell_amount",
    "seat_net_amount",
    "seat_count",
    "seat_concentration",
)

SEAT_E2: tuple[str, ...] = SEAT_AGGREGATE_FEATURES

SEAT_E2_MODEL: tuple[str, ...] = SEAT_AGGREGATE_MODEL_FEATURES

# Northbound detection tokens
_NORTHBOUND_TOKENS = frozenset(
    {"northbound", "north_bound", "沪股通", "深股通", "港资", "北向", "hksc", "hkse"}
)

_PROHIBITED_CONTINUOUS_TOKENS = ("_id", "id_", "hash", "identity")


@dataclass(frozen=True)
class SeatAggregateFeatureRow:
    """Immutable seat aggregate feature row for a stock on a given date."""

    trade_date: str
    stock_id: str
    features: dict[str, float]
    missing_seat_event: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_features(self) -> dict[str, float]:
        """Return B1-ready numeric columns including the explicit missing marker."""
        row = dict(self.features)
        row[MISSING_SEAT_EVENT_FEATURE] = 1.0 if self.missing_seat_event else 0.0
        return row


class SeatAggregateFeatureComputer:
    """Computes stock-level seat aggregate features from DragonTiger records."""

    def __init__(
        self,
        features: Sequence[str] | None = None,
        include_missing_marker: bool = True,
    ) -> None:
        self._features = tuple(features) if features else SEAT_BASELINE
        self._include_missing_marker = include_missing_marker
        self._validate_features()

    def _validate_features(self) -> None:
        """Validate that all requested features are supported."""
        supported = set(SEAT_AGGREGATE_FEATURES)
        for feature in self._features:
            if feature not in supported:
                raise ValueError(f"Unsupported seat aggregate feature: {feature}")

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return ordered feature names."""
        if self._include_missing_marker:
            return (*self._features, MISSING_SEAT_EVENT_FEATURE)
        return self._features

    def compute(
        self,
        records: Sequence[DragonTigerRecord],
        as_of_date: date,
        stock_ids: Sequence[str] | None = None,
        cutoff_time: datetime | None = None,
        historical_records: Sequence[DragonTigerRecord] | None = None,
    ) -> tuple[SeatAggregateFeatureRow, ...]:
        """Compute seat aggregate features for stocks as of a given date.

        Args:
            records: DragonTiger records up to and including as_of_date
            as_of_date: The date to compute features as of
            stock_ids: Optional list of stock IDs to compute features for
            cutoff_time: Optional cutoff time for point-in-time contract
            historical_records: Optional historical records for computing
                               repeat participation features

        Returns:
            Tuple of SeatAggregateFeatureRow for each stock
        """
        # Normalize stock IDs
        target_stocks = {
            normalize_stock_id(sid) for sid in (stock_ids or [])
        }

        # Filter records by date and cutoff_time
        current_records: list[DragonTigerRecord] = []
        for record in records:
            if record.trade_date > as_of_date:
                continue
            if cutoff_time is not None and record.available_time is not None:
                if record.available_time > cutoff_time:
                    continue
            if record.trade_date == as_of_date:
                current_records.append(record)
            target_stocks.add(record.stock_id)

        # Build historical seat lookup for repeat participation
        prior_seats_by_stock: dict[str, set[str]] = defaultdict(set)
        if historical_records:
            for record in historical_records:
                if record.trade_date < as_of_date:
                    prior_seats_by_stock[record.stock_id].add(record.seat_id)

        # Compute features for each stock
        rows: list[SeatAggregateFeatureRow] = []
        for stock_id in sorted(target_stocks):
            stock_records = [r for r in current_records if r.stock_id == stock_id]
            features = _compute_features_for_stock(
                stock_records,
                prior_seats_by_stock.get(stock_id, set()),
            )

            # Filter to requested features
            filtered_features = {
                k: v for k, v in features.items() if k in self._features
            }

            rows.append(
                SeatAggregateFeatureRow(
                    trade_date=as_of_date.isoformat(),
                    stock_id=stock_id,
                    features=filtered_features,
                    missing_seat_event=not stock_records,
                )
            )

        return tuple(rows)


def compute_seat_aggregate_features(
    records: Sequence[DragonTigerRecord],
    as_of_date: date,
    stock_ids: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    cutoff_time: datetime | None = None,
    historical_records: Sequence[DragonTigerRecord] | None = None,
) -> tuple[SeatAggregateFeatureRow, ...]:
    """Convenience function to compute seat aggregate features.

    Args:
        records: DragonTiger records up to as_of_date
        as_of_date: Date to compute features as of
        stock_ids: Optional stock IDs to compute for
        features: Feature names to compute (default: baseline features)
        cutoff_time: Optional cutoff time for PIT contract
        historical_records: Optional historical records

    Returns:
        Tuple of SeatAggregateFeatureRow
    """
    computer = SeatAggregateFeatureComputer(features=features)
    return computer.compute(
        records,
        as_of_date,
        stock_ids=stock_ids,
        cutoff_time=cutoff_time,
        historical_records=historical_records,
    )


def _compute_features_for_stock(
    records: Sequence[DragonTigerRecord],
    prior_seat_ids: set[str],
) -> dict[str, float]:
    """Compute all seat aggregate features for a single stock."""
    features: dict[str, float] = {}

    if not records:
        return _empty_features()

    # Amount aggregations
    buy_records = [r for r in records if r.side == TradeSide.BUY]
    sell_records = [r for r in records if r.side == TradeSide.SELL]

    buy_amount = sum(r.amount for r in buy_records)
    sell_amount = sum(r.amount for r in sell_records)
    net_amount = buy_amount - sell_amount

    # Raw Dragon-Tiger seat amounts reach ~1e10 yuan and cause the E3 seat
    # branch to produce NaN even with LayerNorm after the projection (the
    # projection's weight gradients explode before LayerNorm can clamp the
    # forward pass). Apply log1p on the data side so the model sees values
    # in a bounded range (~0..25) while preserving sign and ordering.
    # sign-preserving log1p: log1p(|x|) with the original sign.
    features["seat_buy_amount"] = _signed_log1p(buy_amount)
    features["seat_sell_amount"] = _signed_log1p(sell_amount)
    features["seat_net_amount"] = _signed_log1p(net_amount)

    # Seat counts
    distinct_seats = {r.seat_id for r in records}
    buy_seats = {r.seat_id for r in buy_records}
    sell_seats = {r.seat_id for r in sell_records}

    features["seat_count"] = float(len(distinct_seats))
    features["seat_buy_count"] = float(len(buy_seats))
    features["seat_sell_count"] = float(len(sell_seats))

    # Institution and northbound ratios
    features["institution_seat_ratio"] = _seat_ratio(
        records, distinct_seats, _is_institution_record
    )
    features["northbound_seat_ratio"] = _seat_ratio(
        records, distinct_seats, _is_northbound_record
    )

    # Concentration (HHI)
    seat_amounts: dict[str, float] = defaultdict(float)
    for record in records:
        seat_amounts[record.seat_id] += record.amount
    total_amount = sum(seat_amounts.values())

    if total_amount > 0:
        hhi = sum((amt / total_amount) ** 2 for amt in seat_amounts.values())
        features["seat_concentration"] = hhi
    else:
        features["seat_concentration"] = 0.0

    # Repeat participation
    if distinct_seats:
        features["seat_repeat_participation"] = (
            len(distinct_seats & prior_seat_ids) / len(distinct_seats)
        )
    else:
        features["seat_repeat_participation"] = 0.0

    # Signal strength (net amount normalized by total)
    if total_amount > 0:
        features["seat_signal_strength"] = net_amount / total_amount
    else:
        features["seat_signal_strength"] = 0.0

    return features


def _empty_features() -> dict[str, float]:
    """Return zero-filled feature dict for stocks with no seat events."""
    return {
        "seat_buy_amount": 0.0,
        "seat_sell_amount": 0.0,
        "seat_net_amount": 0.0,
        "seat_count": 0.0,
        "seat_buy_count": 0.0,
        "seat_sell_count": 0.0,
        "institution_seat_ratio": 0.0,
        "northbound_seat_ratio": 0.0,
        "seat_concentration": 0.0,
        "seat_repeat_participation": 0.0,
        "seat_signal_strength": 0.0,
    }


def _seat_ratio(
    records: Sequence[DragonTigerRecord],
    distinct_seats: set[str],
    predicate: Callable[[DragonTigerRecord], bool],
) -> float:
    """Compute ratio of seats matching a predicate."""
    if not distinct_seats:
        return 0.0
    matching = {
        record.seat_id
        for record in records
        if predicate(record)
    }
    return len(matching & distinct_seats) / len(distinct_seats)


def _is_institution_record(record: DragonTigerRecord) -> bool:
    """Check if record represents an institutional seat."""
    return (
        record.identity_scope == INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE
        or "institution" in record.seat_id.lower()
        or "机构" in record.seat_id
    )


def _is_northbound_record(record: DragonTigerRecord) -> bool:
    """Check if record represents a northbound (HK->CN) seat."""
    text = " ".join(
        item
        for item in (
            record.seat_id,
            record.source_name,
            record.source_version or "",
            record.event_reason or "",
        )
        if item
    ).lower()
    return any(token in text for token in _NORTHBOUND_TOKENS)


def _signed_log1p(value: float) -> float:
    """Sign-preserving log1p for raw monetary seat amounts.

    Maps ~1e10 yuan to ~23.0 while preserving sign and ordering, so the E3
    seat branch receives bounded inputs instead of values that explode the
    projection's weight gradients before LayerNorm can clamp the forward pass.
    NaN/inf inputs collapse to 0.0 so a single bad record cannot poison the
    whole feature vector.
    """
    if not isfinite(value):
        return 0.0
    if value >= 0:
        return log1p(value)
    return -log1p(-value)
