"""Turnover-based portfolio transaction costs shared by candidates and baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCostModel:
    buy_cost_bps: float = 8.0
    sell_cost_bps: float = 18.0

    def __post_init__(self) -> None:
        if self.buy_cost_bps < 0 or self.sell_cost_bps < 0:
            raise ValueError("transaction cost rates must be non-negative")


@dataclass(frozen=True)
class TurnoverCost:
    turnover: float
    buy_turnover: float
    sell_turnover: float
    buy_cost: float
    sell_cost: float

    @property
    def total_cost(self) -> float:
        return self.buy_cost + self.sell_cost


def calculate_turnover_cost(
    current_weights: Mapping[str, float],
    previous_weights: Mapping[str, float],
    *,
    model: TransactionCostModel | None = None,
) -> TurnoverCost:
    model = TransactionCostModel() if model is None else model
    stock_ids = set(current_weights) | set(previous_weights)
    deltas = {
        stock_id: current_weights.get(stock_id, 0.0) - previous_weights.get(stock_id, 0.0)
        for stock_id in stock_ids
    }
    buy_turnover = sum(max(delta, 0.0) for delta in deltas.values())
    sell_turnover = sum(max(-delta, 0.0) for delta in deltas.values())
    return TurnoverCost(
        turnover=0.5 * (buy_turnover + sell_turnover),
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        buy_cost=buy_turnover * model.buy_cost_bps / 10_000.0,
        sell_cost=sell_turnover * model.sell_cost_bps / 10_000.0,
    )


def ranked_portfolio_cost_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    top_k: int,
    date_field: str,
    stock_field: str,
    score_field: str,
    model: TransactionCostModel | None = None,
) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    model = TransactionCostModel() if model is None else model
    by_date: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row[date_field]), []).append(row)
    previous: dict[str, float] = {}
    daily: list[TurnoverCost] = []
    for day in sorted(by_date):
        selected = sorted(
            by_date[day],
            key=lambda row: (-float(row[score_field]), str(row[stock_field])),
        )[:top_k]
        weight = 1.0 / len(selected)
        current = {str(row[stock_field]): weight for row in selected}
        daily.append(calculate_turnover_cost(current, previous, model=model))
        previous = current
    if not daily:
        return {
            "turnover": 0.0,
            "buy_turnover": 0.0,
            "sell_turnover": 0.0,
            "buy_cost": 0.0,
            "sell_cost": 0.0,
            "total_cost": 0.0,
        }
    return {
        name: sum(getattr(item, name) for item in daily) / len(daily)
        for name in ("turnover", "buy_turnover", "sell_turnover", "buy_cost", "sell_cost")
    } | {"total_cost": sum(item.total_cost for item in daily) / len(daily)}
