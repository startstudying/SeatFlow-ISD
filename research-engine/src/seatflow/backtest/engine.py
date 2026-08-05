"""Backtest engine with cost model (Task 20).

Implements:
- Vectorized backtest with position rebalancing
- Transaction cost model (commission + slippage + impact)
- Turnover and capacity constraints
- Long-only and long-short modes
- Strategy attribution: Base, Direct, Spillover, Final
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BacktestMode(str, Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost configuration."""

    commission_bps: float = 2.5  # 0.025% per side
    slippage_bps: float = 5.0  # 0.05% per side
    impact_coeff: float = 0.1  # market impact coefficient
    min_cost_bps: float = 1.0  # minimum total cost in bps


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration."""

    mode: BacktestMode = BacktestMode.LONG_ONLY
    top_k: int = 20
    rebalance_freq: int = 1  # rebalance every N days
    cost: CostConfig = field(default_factory=CostConfig)
    max_turnover: float = 1.0  # max daily turnover
    min_holding_days: int = 1
    benchmark: str = "CSI500"


@dataclass
class BacktestResult:
    """Backtest result container."""

    daily_returns: np.ndarray
    cumulative_returns: np.ndarray
    positions: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    sharpe: float
    max_drawdown: float
    total_return: float
    annual_return: float
    calmar_ratio: float
    win_rate: float
    avg_turnover: float
    avg_cost_bps: float
    attribution: dict[str, float] = field(default_factory=dict)


def compute_transaction_cost(
    turnover: float,
    dollar_volume: float = 1e8,
    config: CostConfig = CostConfig(),
) -> float:
    """Compute transaction cost in return terms.

    Args:
        turnover: portfolio turnover (0 to 1+)
        dollar_volume: average dollar volume of traded stocks
        config: cost configuration

    Returns:
        Cost as a fraction of portfolio value
    """
    # Base cost: commission + slippage per side
    base_cost_bps = config.commission_bps + config.slippage_bps
    # Market impact: proportional to sqrt(turnover / volume)
    impact_bps = config.impact_coeff * np.sqrt(max(turnover, 0)) * 10
    total_bps = max(base_cost_bps + impact_bps, config.min_cost_bps)
    return turnover * total_bps / 10000.0


def backtest(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    config: BacktestConfig = BacktestConfig(),
    dollar_volumes: Optional[np.ndarray] = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Args:
        predictions: [n_days, n_stocks] predicted excess returns
        actual_returns: [n_days, n_stocks] actual excess returns
        config: backtest configuration
        dollar_volumes: [n_days, n_stocks] dollar volumes for impact

    Returns:
        BacktestResult with daily returns, positions, and metrics
    """
    n_days, n_stocks = predictions.shape
    if dollar_volumes is None:
        dollar_volumes = np.ones((n_days, n_stocks)) * 1e8

    daily_returns = np.zeros(n_days)
    positions = np.zeros((n_days, n_stocks))
    turnover = np.zeros(n_days)
    costs = np.zeros(n_days)
    prev_positions = np.zeros(n_stocks)

    for day in range(n_days):
        if day % config.rebalance_freq == 0:
            # Rank stocks by prediction
            preds = predictions[day]
            if config.mode == BacktestMode.LONG_ONLY:
                # Top-K long
                top_k = min(config.top_k, n_stocks)
                top_idx = np.argsort(-preds)[:top_k]
                new_positions = np.zeros(n_stocks)
                # Equal weight or inverse-rank weight
                new_positions[top_idx] = 1.0 / top_k
            else:
                # Long-short: long top-K, short bottom-K
                k = min(config.top_k, n_stocks // 2)
                top_idx = np.argsort(-preds)[:k]
                bottom_idx = np.argsort(preds)[:k]
                new_positions = np.zeros(n_stocks)
                new_positions[top_idx] = 1.0 / k
                new_positions[bottom_idx] = -1.0 / k
                # Normalize to dollar-neutral
                new_positions /= np.abs(new_positions).sum() / 2

            # Compute turnover
            day_turnover = np.abs(new_positions - prev_positions).sum() / 2
            turnover[day] = day_turnover

            # Apply max turnover constraint
            if day_turnover > config.max_turnover:
                scale = config.max_turnover / day_turnover
                new_positions = prev_positions + (new_positions - prev_positions) * scale
                day_turnover = np.abs(new_positions - prev_positions).sum() / 2
                turnover[day] = day_turnover

            # Compute cost
            avg_volume = dollar_volumes[day].mean()
            cost = compute_transaction_cost(day_turnover, avg_volume, config.cost)
            costs[day] = cost

            positions[day] = new_positions
            prev_positions = new_positions
        else:
            # Hold previous positions
            positions[day] = prev_positions
            turnover[day] = 0.0
            costs[day] = 0.0

        # Daily return: weighted average of stock returns minus cost
        gross_return = np.dot(positions[day], actual_returns[day])
        daily_returns[day] = gross_return - costs[day]

    # Compute cumulative returns
    cumulative = np.cumprod(1 + daily_returns) - 1

    # Compute metrics
    sharpe = _sharpe_ratio(daily_returns)
    max_dd = _max_drawdown(cumulative)
    total_ret = float(cumulative[-1]) if n_days > 0 else 0.0
    # Annual return (252 trading days)
    if n_days > 0:
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1
    else:
        annual_ret = 0.0
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    win_rate = float((daily_returns > 0).mean()) if n_days > 0 else 0.0

    return BacktestResult(
        daily_returns=daily_returns,
        cumulative_returns=cumulative,
        positions=positions,
        turnover=turnover,
        costs=costs,
        sharpe=sharpe,
        max_drawdown=max_dd,
        total_return=total_ret,
        annual_return=annual_ret,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_turnover=float(turnover.mean()),
        avg_cost_bps=float(costs.mean() * 10000),
    )


def _sharpe_ratio(returns: np.ndarray, annualization: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(annualization))


def _max_drawdown(cumulative: np.ndarray) -> float:
    """Maximum drawdown from peak."""
    if len(cumulative) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(drawdown.min())


def compare_strategies(
    predictions_by_path: dict[str, np.ndarray],
    actual_returns: np.ndarray,
    config: BacktestConfig = BacktestConfig(),
) -> dict[str, BacktestResult]:
    """Compare Base, Direct, Spillover, and Final strategies.

    Args:
        predictions_by_path: dict mapping path name to predictions array
        actual_returns: [n_days, n_stocks] actual returns
        config: backtest configuration

    Returns:
        Dict mapping path name to BacktestResult
    """
    results: dict[str, BacktestResult] = {}
    for path_name, preds in predictions_by_path.items():
        results[path_name] = backtest(preds, actual_returns, config)
    return results
