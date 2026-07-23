from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal

from seatflow.data.contracts import DailyBar
from seatflow.graph.types import GraphEdge, NodeType


def build_stock_correlation_edges(
    bars: Iterable[DailyBar],
    *,
    as_of_date: date,
    cutoff_time: datetime | None = None,
    lookback_days: int = 60,
    min_overlap: int = 3,
    min_abs_correlation: float = 0.5,
) -> tuple[GraphEdge, ...]:
    if lookback_days <= 1:
        raise ValueError("lookback_days must be greater than one")
    if min_overlap <= 1:
        raise ValueError("min_overlap must be greater than one")
    if not 0 <= min_abs_correlation <= 1:
        raise ValueError("min_abs_correlation must be in [0, 1]")

    returns = _daily_returns(
        bars,
        as_of_date=as_of_date,
        cutoff_time=cutoff_time,
        lookback_days=lookback_days,
    )
    edges: list[GraphEdge] = []
    stock_ids = sorted(returns)
    for left_index, left_stock in enumerate(stock_ids):
        for right_stock in stock_ids[left_index + 1 :]:
            overlap = sorted(set(returns[left_stock]) & set(returns[right_stock]))
            if len(overlap) < min_overlap:
                continue
            left_values = [returns[left_stock][day] for day in overlap]
            right_values = [returns[right_stock][day] for day in overlap]
            correlation = _pearson(left_values, right_values)
            if correlation is None or abs(correlation) < min_abs_correlation:
                continue
            relation = (
                "positively_correlated_with"
                if correlation > 0
                else "negatively_correlated_with"
            )
            for source, target in ((left_stock, right_stock), (right_stock, left_stock)):
                edges.append(
                    GraphEdge(
                        source_type=NodeType.STOCK,
                        source_id=source,
                        relation=relation,
                        target_type=NodeType.STOCK,
                        target_id=target,
                        weight=abs(correlation),
                        event_date=overlap[-1],
                        attributes={
                            "amount": 0.0,
                            "direction": 1.0 if correlation > 0 else -1.0,
                            "delta_days": float((as_of_date - overlap[-1]).days),
                            "time_decay": 1.0,
                            "seat_profile_available": 0.0,
                            "edge_gate_input": abs(correlation),
                            "correlation": correlation,
                            "overlap_days": float(len(overlap)),
                        },
                    )
                )
    return tuple(edges)


def _daily_returns(
    bars: Iterable[DailyBar],
    *,
    as_of_date: date,
    cutoff_time: datetime | None,
    lookback_days: int,
) -> dict[str, dict[date, float]]:
    grouped: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        if bar.trade_date > as_of_date:
            raise ValueError("bars after as_of_date would leak future information")
        if cutoff_time is not None:
            if bar.available_time is None:
                raise ValueError("bar available_time is required when cutoff_time is set")
            _require_compatible_timezone(cutoff_time, bar.available_time)
            if bar.available_time > cutoff_time:
                continue
        grouped[bar.stock_id].append(bar)

    returns: dict[str, dict[date, float]] = {}
    for stock_id, stock_bars in grouped.items():
        ordered = sorted(stock_bars, key=lambda item: item.trade_date)
        recent = ordered[-(lookback_days + 1) :]
        values: dict[date, float] = {}
        previous_close: Decimal | None = None
        for bar in recent:
            if previous_close is not None and previous_close > 0:
                values[bar.trade_date] = float((bar.close_price - previous_close) / previous_close)
            previous_close = bar.close_price
        if values:
            returns[stock_id] = values
    return returns


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0:
        return None
    return numerator / denominator


def _require_compatible_timezone(left: datetime, right: datetime) -> None:
    if (left.tzinfo is None) != (right.tzinfo is None):
        raise ValueError("datetime values must use compatible timezone awareness")
