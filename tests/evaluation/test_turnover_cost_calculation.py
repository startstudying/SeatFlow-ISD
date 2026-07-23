from __future__ import annotations

import unittest

from seatflow.evaluation.portfolio_costs import (
    calculate_turnover_cost,
    ranked_portfolio_cost_summary,
)


class TurnoverCostCalculationTest(unittest.TestCase):
    def test_unchanged_weights_have_zero_cost(self) -> None:
        weights = {"000001.SZ": 0.5, "000002.SZ": 0.5}
        result = calculate_turnover_cost(weights, weights)

        self.assertEqual(result.turnover, 0.0)
        self.assertEqual(result.total_cost, 0.0)

    def test_full_rotation_uses_buy_and_sell_side_rates(self) -> None:
        result = calculate_turnover_cost(
            {"000003.SZ": 0.5, "000004.SZ": 0.5},
            {"000001.SZ": 0.5, "000002.SZ": 0.5},
        )

        self.assertEqual(result.turnover, 1.0)
        self.assertEqual(result.buy_turnover, 1.0)
        self.assertEqual(result.sell_turnover, 1.0)
        self.assertAlmostEqual(result.buy_cost, 0.0008)
        self.assertAlmostEqual(result.sell_cost, 0.0018)
        self.assertAlmostEqual(result.total_cost, 0.0026)

    def test_cost_is_driven_by_daily_weight_change_not_fixed_deduction(self) -> None:
        rows = (
            {"date": "2026-01-01", "stock": "A", "score": 2.0},
            {"date": "2026-01-01", "stock": "B", "score": 1.0},
            {"date": "2026-01-02", "stock": "A", "score": 3.0},
            {"date": "2026-01-02", "stock": "B", "score": 2.0},
        )
        summary = ranked_portfolio_cost_summary(
            rows,
            top_k=2,
            date_field="date",
            stock_field="stock",
            score_field="score",
        )

        self.assertAlmostEqual(summary["total_cost"], 0.0004)
        self.assertAlmostEqual(summary["turnover"], 0.25)


if __name__ == "__main__":
    unittest.main()
