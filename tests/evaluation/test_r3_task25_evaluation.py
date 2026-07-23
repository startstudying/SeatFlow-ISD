from __future__ import annotations

import unittest

from seatflow.evaluation.r3_task25 import (
    _apply_frozen_gates,
    _evaluate_run_daily,
)


class R3Task25EvaluationTest(unittest.TestCase):
    def test_daily_metrics_apply_cost_after_gross_spread(self) -> None:
        predictions = [
            {
                "prediction_date": "2026-01-01",
                "stock_id": f"{index:06d}.SZ",
                "score": float(index),
            }
            for index in range(10)
        ]
        truth = {
            ("2026-01-01", f"{index:06d}.SZ"): float(index) / 100
            for index in range(10)
        }
        market = {
            "2026-01-01": {
                "market_return": 0.01,
                "market_volatility": 0.02,
                "market_amount": 1.0,
                "activity": 10.0,
                "candidate_dispersion": 0.1,
            }
        }

        row = _evaluate_run_daily(predictions, truth, market)[0]

        self.assertGreater(row["top5_gross_spread"], row["top5_net_spread"])
        self.assertEqual(row["turnover"], 0.5)
        self.assertAlmostEqual(row["rank_ic"], 1.0)

    def test_missing_r2_reference_blocks_every_objective(self) -> None:
        template = {
            "run_count": 6,
            "positive_net_runs": 6,
            "fold_net_means": {"0": 0.01, "1": 0.01},
            "maximum_peak_reserved_bytes": 1,
        }
        result = _apply_frozen_gates(
            {name: dict(template) for name in ("L0", "L1", "L2")}
        )

        self.assertEqual(result["status"], "blocked_missing_r2_reference")
        self.assertIsNone(result["selected_objective"])
        self.assertTrue(
            all(not decision["passed"] for decision in result["decisions"].values())
        )


if __name__ == "__main__":
    unittest.main()
