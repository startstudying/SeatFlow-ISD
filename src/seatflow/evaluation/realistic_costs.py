"""Realistic transaction-cost helpers for strategy simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealisticCost:
    """Configurable commission/slippage assumptions."""

    commission: float = 3.0
    stamp: float = 5.0
    buy_slip: float = 5.0
    sell_slip: float = 10.0
    min_commission: float = 5.0
    position_notional: float = 100_000.0


@dataclass(frozen=True)
class RealisticCostComponents:
    """Turnover cost components, all in absolute currency units."""

    turnover: float
    buy_turnover: float
    sell_turnover: float
    commission: float
    stamp: float
    slippage: float
    min_commission: float
    total: float


def effective_side_bps(
    *,
    cost: RealisticCost | None = None,
) -> tuple[float, float]:
    """Return effective buy/sell cost in basis points."""
    if cost is None:
        cost = RealisticCost()
    return (
        (cost.commission + cost.buy_slip) * 1.0,
        (cost.commission + cost.stamp + cost.sell_slip) * 1.0,
    )


def split_turnover_costs(
    *,
    buy_turnover: float,
    sell_turnover: float,
    cost: RealisticCost | None = None,
) -> RealisticCostComponents:
    if buy_turnover < 0 or sell_turnover < 0:
        raise ValueError("turnover values must be non-negative")
    if cost is None:
        cost = RealisticCost()
    if cost.position_notional <= 0:
        raise ValueError("position_notional must be positive")
    if cost.min_commission < 0 or cost.commission < 0 or cost.stamp < 0:
        raise ValueError("non-negative cost assumptions are required")

    turnover = 0.5 * (buy_turnover + sell_turnover)
    buy_notional = buy_turnover * cost.position_notional
    sell_notional = sell_turnover * cost.position_notional
    commission_component = (buy_notional + sell_notional) * cost.commission / 10_000.0
    stamp_component = sell_notional * cost.stamp / 10_000.0
    buy_slippage = buy_notional * cost.buy_slip / 10_000.0
    sell_slippage = sell_notional * cost.sell_slip / 10_000.0
    slippage = buy_slippage + sell_slippage
    raw_total = commission_component + stamp_component + slippage
    min_commission_component = 0.0
    traded_notional = (buy_turnover + sell_turnover) * cost.position_notional
    if traded_notional > 0 and commission_component < cost.min_commission:
        min_commission_component = cost.min_commission - commission_component
    total = raw_total + min_commission_component
    return RealisticCostComponents(
        turnover=turnover,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        commission=commission_component,
        stamp=stamp_component,
        slippage=slippage,
        min_commission=min_commission_component,
        total=total,
    )


def expected_net_utility(
    *,
    expected_return: float,
    buy_turnover: float,
    sell_turnover: float,
    cost: RealisticCost | None = None,
) -> float:
    """Return expected net utility as a decimal portfolio return."""
    active_cost = RealisticCost() if cost is None else cost
    component = split_turnover_costs(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        cost=active_cost,
    )
    return expected_return - component.total / active_cost.position_notional
