from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.labels import StockLabel


@dataclass
class _Accumulator:
    count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    positive_count: int = 0
    signed_return_sum: Decimal = Decimal("0")
    weighted_signed_return_sum: Decimal = Decimal("0")
    amount_sum: float = 0.0
    horizon_sum: int = 0


def build_seat_profile_features(
    records: Iterable[DragonTigerRecord],
    labels: Iterable[StockLabel],
    *,
    as_of_date: date,
    cutoff_time: datetime | None = None,
) -> dict[str, dict[str, float]]:
    """Build leakage-safe soft profiles from matured historical outcomes only."""

    matured = _matured_label_index(labels, as_of_date=as_of_date, cutoff_time=cutoff_time)
    if not matured:
        return {}

    accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)
    for record in records:
        if record.trade_date > as_of_date:
            raise ValueError("records after as_of_date would leak future information")
        if not _record_available(record, cutoff_time):
            continue
        for label in matured.get((record.trade_date, record.stock_id), ()):
            direction = Decimal(1 if record.side == TradeSide.BUY else -1)
            signed_return = label.future_return * direction
            bucket = accumulators[record.seat_id]
            bucket.count += 1
            bucket.buy_count += int(record.side == TradeSide.BUY)
            bucket.sell_count += int(record.side == TradeSide.SELL)
            bucket.positive_count += int(signed_return > 0)
            bucket.signed_return_sum += signed_return
            bucket.weighted_signed_return_sum += signed_return * Decimal(str(record.amount))
            bucket.amount_sum += record.amount
            bucket.horizon_sum += label.horizon

    features: dict[str, dict[str, float]] = {}
    for seat_id, bucket in accumulators.items():
        if bucket.count == 0:
            continue
        weighted = (
            bucket.weighted_signed_return_sum / Decimal(str(bucket.amount_sum))
            if bucket.amount_sum > 0
            else Decimal("0")
        )
        features[seat_id] = {
            "seat_profile_available": 1.0,
            "seat_profile_event_count": float(bucket.count),
            "seat_profile_buy_share": bucket.buy_count / bucket.count,
            "seat_profile_sell_share": bucket.sell_count / bucket.count,
            "seat_profile_positive_rate": bucket.positive_count / bucket.count,
            "seat_profile_avg_signed_return": float(bucket.signed_return_sum / bucket.count),
            "seat_profile_amount_weighted_signed_return": float(weighted),
            "seat_profile_avg_horizon": bucket.horizon_sum / bucket.count,
        }
    return features


def _matured_label_index(
    labels: Iterable[StockLabel],
    *,
    as_of_date: date,
    cutoff_time: datetime | None,
) -> dict[tuple[date, str], tuple[StockLabel, ...]]:
    grouped: dict[tuple[date, str], list[StockLabel]] = defaultdict(list)
    for label in labels:
        if label.as_of_date > as_of_date:
            raise ValueError("labels after as_of_date would leak future information")
        if label.label_start_date <= label.as_of_date:
            raise ValueError("label windows must start after their feature date")
        if cutoff_time is not None:
            _require_compatible_timezone(cutoff_time, label.label_available_time)
            if label.label_available_time > cutoff_time:
                continue
        elif label.label_end_date > as_of_date:
            continue
        grouped[(label.as_of_date, label.target_stock_id)].append(label)
    return {
        key: tuple(sorted(items, key=lambda item: (item.horizon, item.label_end_date)))
        for key, items in grouped.items()
    }


def _record_available(record: DragonTigerRecord, cutoff_time: datetime | None) -> bool:
    if cutoff_time is None:
        return True
    if record.available_time is None:
        raise ValueError("record available_time is required when cutoff_time is set")
    _require_compatible_timezone(cutoff_time, record.available_time)
    return record.available_time <= cutoff_time


def _require_compatible_timezone(left: datetime, right: datetime) -> None:
    if (left.tzinfo is None) != (right.tzinfo is None):
        raise ValueError("datetime values must use compatible timezone awareness")
