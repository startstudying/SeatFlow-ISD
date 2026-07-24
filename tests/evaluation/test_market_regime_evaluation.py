from __future__ import annotations

import unittest

from seatflow.evaluation.regimes import (
    DailyPortfolioMetric,
    evaluate_market_regimes,
)


class MarketRegimeEvaluationTest(unittest.TestCase):
    def test_six_predeclared_groups_emit_complete_metrics(self) -> None:
        rows = tuple(
            DailyPortfolioMetric(
                prediction_date=f"2026-01-0{index + 1}",
                rank_ic=rank_ic,
                top5_gross_spread=gross,
                top5_net_spread=net,
                turnover=turnover,
                market_return=market_return,
                market_volatility=volatility,
            )
            for index, (rank_ic, gross, net, turnover, market_return, volatility) in enumerate(
                (
                    (0.10, 0.02, 0.01, 0.10, 0.01, 0.10),
                    (0.20, 0.03, 0.02, 0.20, 0.02, 0.20),
                    (-0.10, -0.01, -0.02, 0.30, -0.01, 0.30),
                    (0.00, 0.01, 0.00, 0.40, -0.02, 0.40),
                )
            )
        )
        result = evaluate_market_regimes(rows)

        self.assertEqual(
            set(result),
            {
                "up_market",
                "down_market",
                "high_volatility",
                "low_volatility",
                "high_turnover",
                "low_turnover",
            },
        )
        for metrics in result.values():
            self.assertEqual(
                set(metrics),
                {
                    "date_count",
                    "rank_ic",
                    "top5_gross_spread",
                    "top5_net_spread",
                    "positive_return_date_ratio",
                    "turnover",
                },
            )
        self.assertAlmostEqual(result["up_market"]["rank_ic"], 0.15)
        self.assertAlmostEqual(
            result["down_market"]["positive_return_date_ratio"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
