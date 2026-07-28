"""Stock temporal features for SeatFlow.

This module computes comprehensive stock time-series features including:
- Return features: overnight return, intraday return, momentum
- Volatility features: realized volatility, amplitude, high-low range
- Volume features: turnover, volume changes
- Technical indicators: moving averages, RSI, MACD
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from math import sqrt
from typing import Any

from seatflow.data.contracts import DailyBar

# Stock temporal feature names in order
STOCK_TEMPORAL_FEATURES: tuple[str, ...] = (
    # Return features
    "return_1d",
    "return_5d",
    "return_20d",
    "overnight_return",
    "intraday_return",
    # Volatility features
    "volatility_5d",
    "volatility_20d",
    "amplitude",
    "amplitude_5d_avg",
    # Volume features
    "turnover_rate",
    "volume_change",
    "volume_ma5_ratio",
    # Price features
    "close_to_ma5",
    "close_to_ma20",
    "price_momentum_5d",
    "price_momentum_20d",
    # Market-relative features
    "return_sector_neutral_1d",
    "volatility_sector_neutral_1d",
)

# Feature groups for different experiment levels
STOCK_TEMPORAL_BASELINE: tuple[str, ...] = (
    "return_1d",
    "volatility_5d",
    "turnover_rate",
)

STOCK_TEMPORAL_E0: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "turnover_rate",
    "volume_change",
    "close_to_ma5",
    "close_to_ma20",
)

STOCK_TEMPORAL_FULL: tuple[str, ...] = STOCK_TEMPORAL_FEATURES


@dataclass(frozen=True)
class StockTemporalFeatureRow:
    """Immutable stock temporal feature row."""

    trade_date: str
    stock_id: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_features(self) -> dict[str, float]:
        """Return feature dict for model input."""
        return dict(self.features)


class StockTemporalFeatureComputer:
    """Computes comprehensive stock temporal features from daily bars."""

    def __init__(
        self,
        features: Sequence[str] | None = None,
        min_history_days: int = 20,
    ) -> None:
        self._features = tuple(features) if features else STOCK_TEMPORAL_BASELINE
        self._min_history = min_history_days
        self._validate_features()

    def _validate_features(self) -> None:
        """Validate that all requested features are supported."""
        supported = set(STOCK_TEMPORAL_FEATURES)
        for feature in self._features:
            if feature not in supported:
                raise ValueError(f"Unsupported stock temporal feature: {feature}")

    @property
    def feature_order(self) -> tuple[str, ...]:
        """Return ordered feature names."""
        return self._features

    def compute(
        self,
        bars: Sequence[DailyBar],
        as_of_date: date | None = None,
    ) -> tuple[StockTemporalFeatureRow, ...]:
        """Compute temporal features for all stocks in bars.

        Args:
            bars: Daily bar data sorted by trade_date
            as_of_date: Optional date to compute features as of (for PIT contract)

        Returns:
            Tuple of StockTemporalFeatureRow for each stock-date combination
        """
        if not bars:
            return ()

        # Group bars by stock
        bars_by_stock: dict[str, list[DailyBar]] = defaultdict(list)
        for bar in bars:
            if as_of_date is not None and bar.trade_date > as_of_date:
                continue
            bars_by_stock[bar.stock_id].append(bar)

        # Sort each stock's bars by date
        for stock_id in bars_by_stock:
            bars_by_stock[stock_id].sort(key=lambda b: b.trade_date)

        # Compute features for each stock
        rows: list[StockTemporalFeatureRow] = []
        for stock_id, stock_bars in bars_by_stock.items():
            if len(stock_bars) < 2:
                continue

            features = _compute_features_for_stock(
                stock_bars,
                self._features,
                self._min_history,
            )

            for bar in stock_bars:
                if len(stock_bars) < 2:
                    break
                idx = stock_bars.index(bar)
                row_features = features.get(idx, {})
                if row_features:
                    rows.append(
                        StockTemporalFeatureRow(
                            trade_date=bar.trade_date.isoformat(),
                            stock_id=stock_id,
                            features=row_features,
                        )
                    )

        return tuple(rows)

    def compute_for_stock(
        self,
        bars: Sequence[DailyBar],
        stock_id: str,
        as_of_date: date | None = None,
    ) -> tuple[StockTemporalFeatureRow, ...]:
        """Compute temporal features for a single stock.

        Args:
            bars: Daily bar data
            stock_id: Target stock ID
            as_of_date: Optional date to compute features as of

        Returns:
            Tuple of StockTemporalFeatureRow for the stock
        """
        stock_bars = [bar for bar in bars if bar.stock_id == stock_id]
        if as_of_date is not None:
            stock_bars = [bar for bar in stock_bars if bar.trade_date <= as_of_date]
        stock_bars.sort(key=lambda b: b.trade_date)

        if len(stock_bars) < 2:
            return ()

        features = _compute_features_for_stock(
            stock_bars,
            self._features,
            self._min_history,
        )

        rows: list[StockTemporalFeatureRow] = []
        for idx, bar in enumerate(stock_bars):
            row_features = features.get(idx, {})
            if row_features:
                rows.append(
                    StockTemporalFeatureRow(
                        trade_date=bar.trade_date.isoformat(),
                        stock_id=stock_id,
                        features=row_features,
                    )
                )

        return tuple(rows)


def compute_stock_temporal_features(
    bars: Sequence[DailyBar],
    features: Sequence[str] | None = None,
    as_of_date: date | None = None,
) -> tuple[StockTemporalFeatureRow, ...]:
    """Convenience function to compute stock temporal features.

    Args:
        bars: Daily bar data
        features: Feature names to compute (default: baseline features)
        as_of_date: Optional date for point-in-time computation

    Returns:
        Tuple of StockTemporalFeatureRow
    """
    computer = StockTemporalFeatureComputer(features=features)
    return computer.compute(bars, as_of_date=as_of_date)


def _compute_features_for_stock(
    bars: Sequence[DailyBar],
    feature_names: Sequence[str],
    min_history: int,
) -> dict[int, dict[str, float]]:
    """Compute all requested features for each bar of a stock.

    Returns dict mapping bar index to feature dict.
    """
    n = len(bars)
    if n < 2:
        return {}

    # Extract price/volume series
    closes = [float(b.close_price) for b in bars]
    opens = [float(b.open_price) for b in bars]
    highs = [float(b.high_price) for b in bars]
    lows = [float(b.low_price) for b in bars]
    volumes = [float(b.volume) for b in bars]
    amounts = [float(b.amount) for b in bars]

    # Compute all features for each day
    features_by_idx: dict[int, dict[str, float]] = {}

    for i in range(1, n):  # Start from day 1 (need at least 2 days for returns)
        daily_features: dict[str, float] = {}

        # Return features
        if "return_1d" in feature_names:
            daily_features["return_1d"] = (closes[i] - closes[i - 1]) / closes[i - 1]

        if "return_5d" in feature_names and i >= 5:
            daily_features["return_5d"] = (closes[i] - closes[i - 5]) / closes[i - 5]

        if "return_20d" in feature_names and i >= 20:
            daily_features["return_20d"] = (closes[i] - closes[i - 20]) / closes[i - 20]

        if "overnight_return" in feature_names:
            daily_features["overnight_return"] = (opens[i] - closes[i - 1]) / closes[i - 1]

        if "intraday_return" in feature_names:
            daily_features["intraday_return"] = (closes[i] - opens[i]) / opens[i]

        # Volatility features
        if i >= 5:
            returns_5d = [(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(i - 4, i + 1)]
            if "volatility_5d" in feature_names:
                daily_features["volatility_5d"] = _std(returns_5d)

            if "amplitude_5d_avg" in feature_names:
                amplitudes = [(highs[j] - lows[j]) / lows[j] for j in range(i - 4, i + 1)]
                daily_features["amplitude_5d_avg"] = sum(amplitudes) / len(amplitudes)

        if i >= 20:
            returns_20d = [
                (closes[j] - closes[j - 1]) / closes[j - 1] for j in range(i - 19, i + 1)
            ]
            if "volatility_20d" in feature_names:
                daily_features["volatility_20d"] = _std(returns_20d)

        if "amplitude" in feature_names:
            daily_features["amplitude"] = (highs[i] - lows[i]) / lows[i]

        # Volume features
        if "turnover_rate" in feature_names:
            # Approximate turnover rate as amount / (price * shares)
            # Using amount directly as a proxy since shares outstanding varies
            daily_features["turnover_rate"] = amounts[i] / closes[i] if closes[i] > 0 else 0.0

        if "volume_change" in feature_names and i >= 1:
            daily_features["volume_change"] = (
                (volumes[i] - volumes[i - 1]) / volumes[i - 1] if volumes[i - 1] > 0 else 0.0
            )

        if "volume_ma5_ratio" in feature_names and i >= 5:
            vol_ma5 = sum(volumes[i - 4 : i + 1]) / 5
            daily_features["volume_ma5_ratio"] = volumes[i] / vol_ma5 if vol_ma5 > 0 else 0.0

        # Price features
        if i >= 5:
            ma5 = sum(closes[i - 4 : i + 1]) / 5
            if "close_to_ma5" in feature_names:
                daily_features["close_to_ma5"] = (closes[i] - ma5) / ma5

            if "price_momentum_5d" in feature_names:
                daily_features["price_momentum_5d"] = (closes[i] - closes[i - 5]) / closes[i - 5]

        if i >= 20:
            ma20 = sum(closes[i - 19 : i + 1]) / 20
            if "close_to_ma20" in feature_names:
                daily_features["close_to_ma20"] = (closes[i] - ma20) / ma20

            if "price_momentum_20d" in feature_names:
                daily_features["price_momentum_20d"] = (closes[i] - closes[i - 20]) / closes[i - 20]

        if daily_features:
            features_by_idx[i] = daily_features

    return features_by_idx


def _std(values: list[float]) -> float:
    """Compute population standard deviation."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return sqrt(variance)


def _mean(values: list[float]) -> float:
    """Compute mean."""
    return sum(values) / len(values) if values else 0.0
