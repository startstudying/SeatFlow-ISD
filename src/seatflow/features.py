from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal
from math import sqrt
from statistics import fmean

from seatflow.data.contracts import DailyBar, StockFeature


def compute_basic_features(bars: Iterable[DailyBar]) -> tuple[StockFeature, ...]:
    """Compute only trailing features; no record uses a later trading day."""

    by_stock: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        by_stock[bar.stock_id].append(bar)

    features: list[StockFeature] = []
    for stock_id in sorted(by_stock):
        ordered = sorted(by_stock[stock_id], key=lambda item: item.trade_date)
        closes: list[Decimal] = []
        returns: list[Decimal] = []
        for bar in ordered:
            previous_close = closes[-1] if closes else None
            current_return = _return_1d(bar.close_price, previous_close)
            if current_return is not None:
                returns.append(current_return)
            closes.append(bar.close_price)
            features.append(
                StockFeature(
                    trade_date=bar.trade_date,
                    stock_id=stock_id,
                    close_price=bar.close_price,
                    return_1d=current_return,
                    ma_5=_mean_decimal(closes[-5:]) if len(closes) >= 5 else None,
                    ma_20=_mean_decimal(closes[-20:]) if len(closes) >= 20 else None,
                    volatility_5=_population_std(returns[-5:]) if len(returns) >= 5 else None,
                    source_name=bar.source_name,
                    available_time=bar.available_time,
                )
            )
    return tuple(features)


def _return_1d(close_price: Decimal, previous_close: Decimal | None) -> Decimal | None:
    if previous_close is None or previous_close == 0:
        return None
    return (close_price - previous_close) / previous_close


def _mean_decimal(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def _population_std(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("values must not be empty")
    floating = [float(value) for value in values]
    average = fmean(floating)
    return Decimal(str(sqrt(sum((value - average) ** 2 for value in floating) / len(floating))))
