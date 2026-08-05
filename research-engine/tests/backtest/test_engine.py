"""Tests for backtest engine (Task 20)."""

import numpy as np
import pytest

from seatflow.backtest.engine import (
    backtest,
    compare_strategies,
    compute_transaction_cost,
    BacktestConfig,
    BacktestMode,
    CostConfig,
    BacktestResult,
)


class TestTransactionCost:
    def test_zero_turnover_zero_cost(self):
        assert compute_transaction_cost(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_positive_turnover_positive_cost(self):
        cost = compute_transaction_cost(0.5, config=CostConfig(min_cost_bps=0.0))
        assert cost > 0

    def test_higher_turnover_higher_cost(self):
        cost_low = compute_transaction_cost(0.1)
        cost_high = compute_transaction_cost(1.0)
        assert cost_high > cost_low

    def test_min_cost_applied(self):
        """Even with zero turnover, min cost should apply when turnover > 0."""
        config = CostConfig(
            commission_bps=0, slippage_bps=0, impact_coeff=0, min_cost_bps=5.0
        )
        cost = compute_transaction_cost(0.5, config=config)
        # 0.5 * 5.0 / 10000
        assert cost == pytest.approx(0.5 * 5.0 / 10000.0)


class TestBacktest:
    def test_shape_preserved(self):
        n_days, n_stocks = 10, 5
        preds = np.random.randn(n_days, n_stocks)
        returns = np.random.randn(n_days, n_stocks) * 0.01
        result = backtest(preds, returns)
        assert result.daily_returns.shape == (n_days,)
        assert result.positions.shape == (n_days, n_stocks)
        assert result.turnover.shape == (n_days,)

    def test_no_cost_when_no_rebalancing(self):
        """If rebalance_freq > n_days, only one rebalance occurs."""
        preds = np.ones((5, 3))
        returns = np.zeros((5, 3))
        config = BacktestConfig(rebalance_freq=10, top_k=3)
        result = backtest(preds, returns, config)
        # Only day 0 has turnover
        assert result.turnover[0] > 0
        assert result.turnover[1:].sum() == pytest.approx(0.0)

    def test_cost_subtracted_from_return(self):
        """Daily return must be gross return minus cost."""
        preds = np.array([[1.0, 0.0, -1.0]] * 5)
        returns = np.array([[0.01, 0.0, -0.01]] * 5)
        config = BacktestConfig(top_k=1, cost=CostConfig(commission_bps=10, slippage_bps=10))
        result = backtest(preds, returns, config)
        # Day 0: gross = 0.01 (long stock 0), cost > 0
        assert result.costs[0] > 0
        assert result.daily_returns[0] < 0.01  # gross minus cost

    def test_long_short_mode(self):
        preds = np.array([[1.0, 0.0, -1.0]] * 5)
        returns = np.array([[0.01, 0.0, -0.01]] * 5)
        config = BacktestConfig(mode=BacktestMode.LONG_SHORT, top_k=1)
        result = backtest(preds, returns, config)
        # Long stock 0 (return +0.01), short stock 2 (return +0.01 since -1 * -0.01)
        # Gross = 0.01 + 0.01 = 0.02
        assert result.daily_returns[1] > 0

    def test_turnover_constraint(self):
        """Turnover should not exceed max_turnover."""
        preds = np.random.randn(10, 5)
        returns = np.zeros((10, 5))
        config = BacktestConfig(max_turnover=0.1, top_k=5)
        result = backtest(preds, returns, config)
        assert (result.turnover <= 0.1 + 1e-6).all()

    def test_max_drawdown_non_positive(self):
        preds = np.random.randn(20, 5)
        returns = np.random.randn(20, 5) * 0.01
        result = backtest(preds, returns)
        assert result.max_drawdown <= 0.0

    def test_sharpe_finite(self):
        preds = np.random.randn(20, 5)
        returns = np.random.randn(20, 5) * 0.01
        result = backtest(preds, returns)
        assert np.isfinite(result.sharpe)


class TestStrategyComparison:
    def test_compare_four_paths(self):
        n_days, n_stocks = 20, 10
        actual = np.random.randn(n_days, n_stocks) * 0.01
        preds_by_path = {
            "Base": np.random.randn(n_days, n_stocks),
            "Direct": np.random.randn(n_days, n_stocks),
            "Spillover": np.random.randn(n_days, n_stocks),
            "Final": np.random.randn(n_days, n_stocks),
        }
        results = compare_strategies(preds_by_path, actual)
        assert len(results) == 4
        for name, result in results.items():
            assert isinstance(result, BacktestResult)
            assert result.daily_returns.shape == (n_days,)

    def test_final_outperforms_base_with_good_predictions(self):
        """If Final predictions are perfect, it should beat random Base."""
        rng = np.random.default_rng(42)
        n_days, n_stocks = 20, 10
        actual = rng.standard_normal((n_days, n_stocks)) * 0.02
        # Final uses actual returns as predictions (perfect)
        # Base uses random predictions (different from actual)
        base_preds = rng.standard_normal((n_days, n_stocks))
        # Use top_k=3 so that selection actually matters
        config = BacktestConfig(top_k=3)
        results = compare_strategies(
            {
                "Base": base_preds,
                "Final": actual.copy(),  # perfect prediction
            },
            actual,
            config=config,
        )
        assert results["Final"].total_return > results["Base"].total_return
