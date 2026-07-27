"""Point-in-time stock-level R4 seat aggregate features.

R4 intentionally represents seat information as stock-node aggregate columns,
not as Seat nodes, Seat-Stock edges, Seat-Seat co-occurrence, or numeric
identity hashes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite, sqrt

from seatflow.data.contracts import normalize_stock_id
from seatflow.data.records import (
    INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    DragonTigerRecord,
    TradeSide,
)
from seatflow.r4.decayed_seat_state import DecayedSeatState

SEAT_AGGREGATE_FEATURES: tuple[str, ...] = (
    "seat_buy_amount",
    "seat_sell_amount",
    "seat_net_amount",
    "seat_count",
    "institution_seat_ratio",
    "northbound_seat_ratio",
    "seat_concentration",
    "seat_repeat_participation",
    "days_since_last_event",
    "decayed_seat_signal",
)
MISSING_SEAT_EVENT_FEATURE = "missing_seat_event"
SEAT_AGGREGATE_MODEL_FEATURES: tuple[str, ...] = (
    *SEAT_AGGREGATE_FEATURES,
    MISSING_SEAT_EVENT_FEATURE,
)

B0_FEATURE_COLUMN_PROTOCOL: tuple[str, ...] = (
    "stock_price_features",
    "industry_features",
    "market_state_features",
)
B1_FEATURE_COLUMN_PROTOCOL: tuple[str, ...] = (
    *B0_FEATURE_COLUMN_PROTOCOL,
    *SEAT_AGGREGATE_MODEL_FEATURES,
)

_TRANSFORMED_SENTINEL = "__seat_aggregate_scaled__"
_PROHIBITED_CONTINUOUS_TOKENS = ("_id", "id_", "hash", "identity")
_NORTHBOUND_TOKENS = ("northbound", "north_bound", "沪股通", "深股通", "港资", "北向")


class SeatAggregateFeatureError(ValueError):
    """Raised when R4 seat aggregate features would violate the PIT contract."""


@dataclass(frozen=True)
class SeatAggregateFeatureRow:
    trade_date: str
    stock_id: str
    features: dict[str, float]
    missing_seat_event: bool

    def __post_init__(self) -> None:
        normalized_features = _validated_feature_row(
            self.features,
            SEAT_AGGREGATE_FEATURES,
            allow_missing_marker=False,
        )
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        object.__setattr__(self, "features", normalized_features)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def model_features(self) -> dict[str, float]:
        """Return B1-ready numeric columns including the explicit missing marker."""

        row = dict(self.features)
        row[MISSING_SEAT_EVENT_FEATURE] = 1.0 if self.missing_seat_event else 0.0
        return row


class TrainOnlyStandardScaler:
    """Deterministic scaler fitted once on training rows, then reused unchanged."""

    def __init__(self, feature_order: Sequence[str] = SEAT_AGGREGATE_MODEL_FEATURES) -> None:
        self._feature_order = _validate_feature_order(feature_order)
        self._means: dict[str, float] | None = None
        self._scales: dict[str, float] | None = None

    @property
    def fitted(self) -> bool:
        return self._means is not None and self._scales is not None

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self._feature_order

    def fit(self, rows: Sequence[Mapping[str, float]]) -> None:
        if self.fitted:
            raise RuntimeError("scaler has already been fitted")
        if not rows:
            raise ValueError("cannot fit scaler on empty training rows")
        values_by_feature: dict[str, list[float]] = {feature: [] for feature in self.feature_order}
        for row in rows:
            validated = _validated_feature_row(
                row,
                self.feature_order,
                allow_missing_marker=MISSING_SEAT_EVENT_FEATURE in self.feature_order,
            )
            for feature in self.feature_order:
                values_by_feature[feature].append(validated[feature])
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature, values in values_by_feature.items():
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[feature] = mean
            scales[feature] = sqrt(variance) or 1.0
        self._means = means
        self._scales = scales

    def transform(self, rows: Sequence[Mapping[str, float]]) -> tuple[dict[str, float], ...]:
        if self._means is None or self._scales is None:
            raise RuntimeError("fit scaler on training rows before transform")
        transformed: list[dict[str, float]] = []
        for row in rows:
            if _TRANSFORMED_SENTINEL in row:
                raise RuntimeError("seat aggregate row has already been transformed")
            validated = _validated_feature_row(
                row,
                self.feature_order,
                allow_missing_marker=MISSING_SEAT_EVENT_FEATURE in self.feature_order,
            )
            output = {
                feature: (validated[feature] - self._means[feature]) / self._scales[feature]
                for feature in self.feature_order
            }
            output[_TRANSFORMED_SENTINEL] = 1.0
            transformed.append(output)
        return tuple(transformed)


def build_stock_seat_aggregate_features_asof(
    *,
    records: Sequence[DragonTigerRecord],
    as_of_date: date,
    stock_ids: Sequence[str],
    decayed_states: Sequence[DecayedSeatState] = (),
    cutoff_time: datetime | None = None,
) -> tuple[SeatAggregateFeatureRow, ...]:
    """Build stock-level aggregate columns from records visible at ``as_of_date`` only."""

    normalized_stock_ids = {normalize_stock_id(stock_id) for stock_id in stock_ids}
    current_by_stock: dict[str, list[DragonTigerRecord]] = defaultdict(list)
    historical_seats_by_stock: dict[str, set[str]] = defaultdict(set)
    latest_event_date_by_stock: dict[str, date] = {}
    for record in records:
        if record.trade_date > as_of_date:
            raise SeatAggregateFeatureError(
                "records after as_of_date would leak future information"
            )
        if cutoff_time is not None:
            if record.available_time is None:
                raise SeatAggregateFeatureError(
                    "record available_time is required when cutoff_time is set"
                )
            if (record.available_time.tzinfo is None) != (cutoff_time.tzinfo is None):
                raise SeatAggregateFeatureError(
                    "record and cutoff_time must use compatible timezone awareness"
                )
            if record.available_time > cutoff_time:
                continue
        normalized_stock_ids.add(record.stock_id)
        if record.trade_date < as_of_date:
            historical_seats_by_stock[record.stock_id].add(record.seat_id)
            previous = latest_event_date_by_stock.get(record.stock_id)
            if previous is None or record.trade_date > previous:
                latest_event_date_by_stock[record.stock_id] = record.trade_date
        elif record.trade_date == as_of_date:
            current_by_stock[record.stock_id].append(record)
            latest_event_date_by_stock[record.stock_id] = record.trade_date

    decayed_by_stock = _decayed_states_by_stock(decayed_states, as_of_date)
    rows: list[SeatAggregateFeatureRow] = []
    for stock_id in sorted(normalized_stock_ids):
        current_records = tuple(current_by_stock.get(stock_id, ()))
        decayed_state = decayed_by_stock.get(stock_id)
        features = _features_for_stock(
            stock_id=stock_id,
            current_records=current_records,
            prior_seat_ids=historical_seats_by_stock.get(stock_id, set()),
            decayed_state=decayed_state,
            latest_event_date=latest_event_date_by_stock.get(stock_id),
            as_of_date=as_of_date,
        )
        rows.append(
            SeatAggregateFeatureRow(
                trade_date=as_of_date.isoformat(),
                stock_id=stock_id,
                features=features,
                missing_seat_event=not current_records,
            )
        )
    return tuple(rows)


def aggregate_seat_features_asof(
    *,
    trade_date: str,
    stock_ids: Sequence[str],
    event_features_by_stock: Mapping[str, Mapping[str, float]],
    decayed_signals_by_stock: Mapping[str, float],
) -> tuple[SeatAggregateFeatureRow, ...]:
    """Compatibility adapter for callers that already have PIT event aggregates."""

    rows: list[SeatAggregateFeatureRow] = []
    for stock_id in sorted({normalize_stock_id(item) for item in stock_ids}):
        event = event_features_by_stock.get(stock_id)
        missing = event is None
        features: dict[str, float] = {}
        for feature in SEAT_AGGREGATE_FEATURES:
            if feature == "decayed_seat_signal":
                features[feature] = float(decayed_signals_by_stock.get(stock_id, 0.0))
            elif event is None:
                features[feature] = 0.0
            else:
                features[feature] = float(event.get(feature, 0.0))
        rows.append(
            SeatAggregateFeatureRow(
                trade_date=trade_date,
                stock_id=stock_id,
                features=features,
                missing_seat_event=missing,
            )
        )
    return tuple(rows)


def _features_for_stock(
    *,
    stock_id: str,
    current_records: Sequence[DragonTigerRecord],
    prior_seat_ids: set[str],
    decayed_state: DecayedSeatState | None,
    latest_event_date: date | None,
    as_of_date: date,
) -> dict[str, float]:
    buy_amount = sum(record.amount for record in current_records if record.side == TradeSide.BUY)
    sell_amount = sum(record.amount for record in current_records if record.side == TradeSide.SELL)
    distinct_seats = {record.seat_id for record in current_records}
    seat_amounts: dict[str, float] = defaultdict(float)
    for record in current_records:
        seat_amounts[record.seat_id] += record.amount
    total_amount = sum(seat_amounts.values())
    seat_count = len(distinct_seats)
    features = {
        "seat_buy_amount": buy_amount,
        "seat_sell_amount": sell_amount,
        "seat_net_amount": buy_amount - sell_amount,
        "seat_count": float(seat_count),
        "institution_seat_ratio": _seat_ratio(current_records, _is_institution_record),
        "northbound_seat_ratio": _seat_ratio(current_records, _is_northbound_record),
        "seat_concentration": (
            sum((amount / total_amount) ** 2 for amount in seat_amounts.values())
            if total_amount > 0.0
            else 0.0
        ),
        "seat_repeat_participation": (
            len(distinct_seats & prior_seat_ids) / seat_count if seat_count else 0.0
        ),
        "days_since_last_event": _days_since_last_event(
            stock_id=stock_id,
            as_of_date=as_of_date,
            decayed_state=decayed_state,
            latest_event_date=latest_event_date,
        ),
        "decayed_seat_signal": (
            decayed_state.decayed_seat_signal
            if decayed_state is not None
            else buy_amount - sell_amount
        ),
    }
    return _validated_feature_row(features, SEAT_AGGREGATE_FEATURES, allow_missing_marker=False)


def _decayed_states_by_stock(
    states: Sequence[DecayedSeatState], as_of_date: date
) -> dict[str, DecayedSeatState]:
    target_date = as_of_date.isoformat()
    by_stock: dict[str, DecayedSeatState] = {}
    for state in states:
        if state.trade_date != target_date:
            continue
        by_stock[normalize_stock_id(state.stock_id)] = state
    return by_stock


def _days_since_last_event(
    *,
    stock_id: str,
    as_of_date: date,
    decayed_state: DecayedSeatState | None,
    latest_event_date: date | None,
) -> float:
    if decayed_state is not None:
        return float(decayed_state.days_since_last_seat_event or 0)
    if latest_event_date is None:
        return 0.0
    if latest_event_date > as_of_date:
        raise SeatAggregateFeatureError(f"latest event for {stock_id} is after as_of_date")
    return float((as_of_date - latest_event_date).days)


def _seat_ratio(
    records: Sequence[DragonTigerRecord],
    predicate: Callable[[DragonTigerRecord], bool],
) -> float:
    seats = {record.seat_id for record in records}
    if not seats:
        return 0.0
    matching = {
        record.seat_id
        for record in records
        if predicate(record)
    }
    return len(matching) / len(seats)


def _is_institution_record(record: DragonTigerRecord) -> bool:
    return (
        record.identity_scope == INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE
        or "institution" in record.seat_id.lower()
        or "机构" in record.seat_id
    )


def _is_northbound_record(record: DragonTigerRecord) -> bool:
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


def _validate_feature_order(feature_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(feature_order)
    if not order or len(set(order)) != len(order):
        raise SeatAggregateFeatureError("feature order must be non-empty and unique")
    for feature in order:
        normalized = feature.lower()
        if any(token in normalized for token in _PROHIBITED_CONTINUOUS_TOKENS):
            raise SeatAggregateFeatureError("identity/hash features cannot be continuous columns")
    return order


def _validated_feature_row(
    row: Mapping[str, float],
    feature_order: Sequence[str],
    *,
    allow_missing_marker: bool,
) -> dict[str, float]:
    order = _validate_feature_order(feature_order)
    expected = set(order)
    if allow_missing_marker:
        expected.add(MISSING_SEAT_EVENT_FEATURE)
    if set(row) != expected:
        if _TRANSFORMED_SENTINEL in row:
            raise SeatAggregateFeatureError("seat aggregate row has already been transformed")
        raise SeatAggregateFeatureError("row features do not exactly match seat aggregate schema")
    validated: dict[str, float] = {}
    for feature in order:
        value = float(row[feature])
        if not isfinite(value):
            raise SeatAggregateFeatureError(f"feature {feature!r} is non-finite")
        validated[feature] = value
    if allow_missing_marker:
        marker = float(row[MISSING_SEAT_EVENT_FEATURE])
        if marker not in {0.0, 1.0}:
            raise SeatAggregateFeatureError("missing_seat_event must be an explicit 0/1 marker")
        validated[MISSING_SEAT_EVENT_FEATURE] = marker
    return validated
