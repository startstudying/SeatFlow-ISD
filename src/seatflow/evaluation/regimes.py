"""Market-regime diagnostics for R2 portfolio stability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class DailyPortfolioMetric:
    prediction_date: str
    rank_ic: float
    top5_gross_spread: float
    top5_net_spread: float
    turnover: float
    market_return: float
    market_volatility: float


def evaluate_market_regimes(
    rows: Sequence[DailyPortfolioMetric],
) -> dict[str, dict[str, float | int | None]]:
    if not rows:
        raise ValueError("market regime evaluation requires daily rows")
    volatility_median = median(row.market_volatility for row in rows)
    turnover_median = median(row.turnover for row in rows)
    groups = {
        "up_market": [row for row in rows if row.market_return >= 0],
        "down_market": [row for row in rows if row.market_return < 0],
        "high_volatility": [
            row for row in rows if row.market_volatility > volatility_median
        ],
        "low_volatility": [
            row for row in rows if row.market_volatility <= volatility_median
        ],
        "high_turnover": [row for row in rows if row.turnover > turnover_median],
        "low_turnover": [row for row in rows if row.turnover <= turnover_median],
    }
    return {name: _summarize(group) for name, group in groups.items()}


def _summarize(rows: Sequence[DailyPortfolioMetric]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "date_count": 0,
            "rank_ic": None,
            "top5_gross_spread": None,
            "top5_net_spread": None,
            "positive_return_date_ratio": None,
            "turnover": None,
        }
    count = len(rows)
    return {
        "date_count": count,
        "rank_ic": sum(row.rank_ic for row in rows) / count,
        "top5_gross_spread": sum(row.top5_gross_spread for row in rows) / count,
        "top5_net_spread": sum(row.top5_net_spread for row in rows) / count,
        "positive_return_date_ratio": (
            sum(row.top5_net_spread > 0 for row in rows) / count
        ),
        "turnover": sum(row.turnover for row in rows) / count,
    }
