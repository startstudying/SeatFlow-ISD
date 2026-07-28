"""Market environment features for SeatFlow.

This module computes market-wide environment features including:
- CSI300 index returns and volatility
- Market median returns
- Cross-sectional spread features
- Market regime indicators
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from math import sqrt
from typing import Any

from seatflow.data.contracts import DailyBar

# Market environment feature names
MARKET_ENV_FEATURES: tuple[str, ...] = (
    # Index features
    "index_return_1d",
    "index_return_5d",
    "index_volatility_5d",
    "index_volatility_20d",
    # Market-wide features
    "market_median_return_1d",
    "market_spread_return_1d",
    "market_up_count_ratio",
    "market_avg_volume_change",
    # Regime indicators
    "market_regime_volatility",
    "market_regime_trend",
)

# Feature groups by experiment level
MARKET_BASELINE: tuple[str, ...] = (
    "index_return_1d",
    "market_median_return_1d",
)

MARKET_E0: tuple[str, ...] = (
    "index_return_1d",
    "index_return_5d",
    "index_volatility_5d",
    "market_median_return_1d",
    "market_spread_return_1d",
)

MARKET_FULL: tuple[str, ...] = MARKET_ENV_FEATURES

# CSI300 component codes (approximate - in practice would come from config)
_CSI300_CODES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MarketEnvFeatureRow:
    """Immutable market environment feature row for a given date."""

    trade_date: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_features(self) -> dict[str, float]:
        """Return feature dict for model input."""
        return dict(self.features)


@dataclass(frozen=True)
class StockMarketFeatureRow:
    """Stock-level market-relative features."""

    trade_date: str
    stock_id: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_features(self) -> dict[str, float]:
        """Return feature dict for model input."""
        return dict(self.features)


class MarketEnvFeatureComputer:
    """Computes market environment features from daily bars."""

    def __init__(
        self,
        features: Sequence[str] | None = None,
        index_codes: Sequence[str] | None = None,
        min_stocks_for_market: int = 50,
    ) -> None:
        self._features = tuple(features) if features else MARKET_BASELINE
        self._index_codes = frozenset(index_codes or [])
        self._min_stocks = min_stocks_for_market
        self._validate_features()

    def _validate_features(self) -> None:
        """Validate that all requested features are supported."""
        supported = set(MARKET_ENV_FEATURES)
        for feature in self._features:
            if feature not in supported:
                raise ValueError(f"Unsupported market env feature: {feature}")

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return ordered feature names."""
        return self._features

    def compute_market_features(
        self,
        bars: Sequence[DailyBar],
        as_of_date: date | None = None,
    ) -> tuple[MarketEnvFeatureRow, ...]:
        """Compute market-level features from all available bars.

        Args:
            bars: Daily bar data
            as_of_date: Optional date to compute features as of

        Returns:
            Tuple of MarketEnvFeatureRow for each date
        """
        # Group bars by date
        bars_by_date: dict[date, list[DailyBar]] = defaultdict(list)
        for bar in bars:
            if as_of_date is not None and bar.trade_date > as_of_date:
                continue
            bars_by_date[bar.trade_date].append(bar)

        # Compute features for each date
        rows: list[MarketEnvFeatureRow] = []
        for trade_date in sorted(bars_by_date.keys()):
            date_bars = bars_by_date[trade_date]
            if len(date_bars) < self._min_stocks:
                continue

            features = _compute_market_features_for_date(
                trade_date,
                date_bars,
                bars_by_date,
                self._features,
                self._index_codes,
            )

            if features:
                rows.append(
                    MarketEnvFeatureRow(
                        trade_date=trade_date.isoformat(),
                        features=features,
                    )
                )

        return tuple(rows)

    def compute_stock_market_features(
        self,
        bars: Sequence[DailyBar],
        as_of_date: date | None = None,
    ) -> tuple[StockMarketFeatureRow, ...]:
        """Compute stock-level market-relative features.

        Args:
            bars: Daily bar data
            as_of_date: Optional date to compute features as of

        Returns:
            Tuple of StockMarketFeatureRow for each stock-date
        """
        # Group bars by date and stock
        bars_by_date: dict[date, list[DailyBar]] = defaultdict(list)
        bars_by_stock: dict[str, list[DailyBar]] = defaultdict(list)

        for bar in bars:
            if as_of_date is not None and bar.trade_date > as_of_date:
                continue
            bars_by_date[bar.trade_date].append(bar)
            bars_by_stock[bar.stock_id].append(bar)

        # Sort stock bars by date
        for stock_id in bars_by_stock:
            bars_by_stock[stock_id].sort(key=lambda b: b.trade_date)

        # Compute stock-level market-relative features
        rows: list[StockMarketFeatureRow] = []
        for stock_id, stock_bars in bars_by_stock.items():
            if len(stock_bars) < 2:
                continue

            for i, bar in enumerate(stock_bars):
                if as_of_date is not None and bar.trade_date > as_of_date:
                    continue

                features = _compute_stock_market_features(
                    bar,
                    i,
                    stock_bars,
                    bars_by_date,
                    self._index_codes,
                )

                if features:
                    rows.append(
                        StockMarketFeatureRow(
                            trade_date=bar.trade_date.isoformat(),
                            stock_id=stock_id,
                            features=features,
                        )
                    )

        return tuple(rows)


def compute_market_env_features(
    bars: Sequence[DailyBar],
    features: Sequence[str] | None = None,
    as_of_date: date | None = None,
) -> tuple[MarketEnvFeatureRow, ...]:
    """Convenience function to compute market environment features.

    Args:
        bars: Daily bar data
        features: Feature names to compute (default: baseline features)
        as_of_date: Optional date for point-in-time computation

    Returns:
        Tuple of MarketEnvFeatureRow
    """
    computer = MarketEnvFeatureComputer(features=features)
    return computer.compute_market_features(bars, as_of_date=as_of_date)


def _compute_market_features_for_date(
    trade_date: date,
    date_bars: list[DailyBar],
    bars_by_date: Mapping[date, list[DailyBar]],
    feature_names: tuple[str, ...],
    index_codes: frozenset[str],
) -> dict[str, float]:
    """Compute all market features for a single date."""
    features: dict[str, float] = {}

    if not date_bars:
        return features

    # Extract stock returns for this date
    stock_returns: dict[str, float] = {}
    for bar in date_bars:
        prev_bars = [b for b in bars_by_date.get(trade_date, []) if b.stock_id == bar.stock_id]
        if len(prev_bars) < 2:
            # Look for previous day
            prev_dates = sorted(bars_by_date.keys())
            prev_idx = prev_dates.index(trade_date) - 1 if trade_date in prev_dates else -1
            if prev_idx >= 0:
                prev_date = prev_dates[prev_idx]
                prev_bar_list = [b for b in bars_by_date[prev_date] if b.stock_id == bar.stock_id]
                if prev_bar_list:
                    prev_close = float(prev_bar_list[0].close_price)
                    curr_close = float(bar.close_price)
                    if prev_close > 0:
                        stock_returns[bar.stock_id] = (curr_close - prev_close) / prev_close

    if not stock_returns:
        return features

    returns = list(stock_returns.values())
    market_return = sum(returns) / len(returns) if returns else 0.0

    # Index features (if index codes provided)
    if index_codes:
        index_returns = [r for sid, r in stock_returns.items() if sid in index_codes]
        if index_returns:
            features["index_return_1d"] = sum(index_returns) / len(index_returns)
    else:
        # Use market median as index proxy
        features["index_return_1d"] = market_return

    # 5-day index return
    if "index_return_5d" in feature_names:
        prev_dates = sorted(bars_by_date.keys())
        curr_idx = prev_dates.index(trade_date) if trade_date in prev_dates else -1
        if curr_idx >= 5:
            five_day_returns = []
            for d in prev_dates[curr_idx - 4 : curr_idx + 1]:
                day_bars = bars_by_date.get(d, [])
                day_returns = []
                for bar in day_bars:
                    prev_day_bars = bars_by_date.get(prev_dates[prev_dates.index(d) - 1], [])
                    prev_bar = [b for b in prev_day_bars if b.stock_id == bar.stock_id]
                    if prev_bar:
                        prev_close = float(prev_bar[0].close_price)
                        curr_close = float(bar.close_price)
                        if prev_close > 0:
                            day_returns.append((curr_close - prev_close) / prev_close)
                if day_returns:
                    five_day_returns.append(sum(day_returns) / len(day_returns))
            if len(five_day_returns) >= 5:
                cum_return = 1.0
                for r in five_day_returns:
                    cum_return *= (1 + r)
                features["index_return_5d"] = cum_return - 1.0

    # Volatility features
    if "index_volatility_5d" in feature_names and len(returns) >= 5:
        features["index_volatility_5d"] = _std(returns)
    if "index_volatility_20d" in feature_names and len(returns) >= 20:
        features["index_volatility_20d"] = _std(returns)

    # Market-wide features
    if returns:
        sorted_returns = sorted(returns)
        features["market_median_return_1d"] = sorted_returns[len(sorted_returns) // 2]

        if len(sorted_returns) > 1:
            features["market_spread_return_1d"] = sorted_returns[-1] - sorted_returns[0]

        up_count = sum(1 for r in returns if r > 0)
        features["market_up_count_ratio"] = up_count / len(returns)

    # Only emit requested features so the output schema matches feature_order.
    # Without this filter, compute_market_features emits columns the caller did
    # not request, breaking the shape contract (len(row.features) must equal
    # len(feature_order)) and leaking undeclared features downstream.
    if feature_names:
        features = {name: value for name, value in features.items() if name in feature_names}

    return features


def _compute_stock_market_features(
    bar: DailyBar,
    bar_idx: int,
    stock_bars: list[DailyBar],
    bars_by_date: Mapping[date, list[DailyBar]],
    index_codes: frozenset[str],
) -> dict[str, float]:
    """Compute market-relative features for a single stock."""
    features: dict[str, float] = {}

    if bar_idx < 1:
        return features

    # Get previous bar for return calculation
    prev_bar = stock_bars[bar_idx - 1]
    stock_return = 0.0
    if float(prev_bar.close_price) > 0:
        stock_return = (
            float(bar.close_price) - float(prev_bar.close_price)
        ) / float(prev_bar.close_price)

    # Get market return for this date
    date_bars = bars_by_date.get(bar.trade_date, [])
    market_returns = []
    for b in date_bars:
        if b.stock_id == bar.stock_id:
            continue
        prev_bars = [bb for bb in bars_by_date.get(bar.trade_date, []) if bb.stock_id == b.stock_id]
        if len(prev_bars) >= 2:
            prev_b = prev_bars[0]
            if float(prev_b.close_price) > 0:
                market_returns.append(
                    (float(b.close_price) - float(prev_b.close_price)) / float(prev_b.close_price)
                )

    if market_returns:
        market_return = sum(market_returns) / len(market_returns)
        features["return_sector_neutral_1d"] = stock_return - market_return
    else:
        features["return_sector_neutral_1d"] = 0.0

    return features


def _std(values: list[float]) -> float:
    """Compute population standard deviation."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return sqrt(variance)
