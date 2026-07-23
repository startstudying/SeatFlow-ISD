from __future__ import annotations

import unittest

from seatflow.evaluation.stability import (
    PredictionRow,
    RunMetric,
    analyze_stability,
    jaccard,
    pearson,
    spearman,
)


class R2StabilityTest(unittest.TestCase):
    def test_core_metrics(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 3, 2], [10, 30, 20]), 1.0)
        self.assertAlmostEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)

    def test_incomplete_evidence_fails_closed_without_classification(self) -> None:
        result = analyze_stability(
            [
                PredictionRow(
                    fold=0,
                    seed=2026,
                    prediction_date="2026-01-01",
                    stock_id="000001.SZ",
                    score=1.0,
                    target=0.1,
                )
            ],
            [
                RunMetric(
                    fold=0,
                    seed=2026,
                    rank_ic=0.1,
                    gross_spread=0.01,
                    net_spread=0.0,
                    turnover=0.5,
                )
            ],
        )

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["failure_classes"], [])
        self.assertTrue(any("missing_run_metrics" in gap for gap in result["evidence_gaps"]))
        self.assertIn("missing_inference_ablations", result["evidence_gaps"])
        self.assertIn("missing_market_regimes", result["evidence_gaps"])

    def test_complete_matrix_produces_fold_and_regime_metrics(self) -> None:
        predictions = [
            PredictionRow(
                fold=fold,
                seed=seed,
                prediction_date="2026-01-01",
                stock_id=f"{stock:06d}.SZ",
                score=float(stock) + seed * 1e-6,
                target=float(stock),
            )
            for fold in range(3)
            for seed in range(2026, 2031)
            for stock in range(10)
        ]
        metrics = [
            RunMetric(
                fold=fold,
                seed=seed,
                rank_ic=0.05,
                gross_spread=0.01,
                net_spread=0.005,
                turnover=0.2,
                regime="up" if fold < 2 else "down",
            )
            for fold in range(3)
            for seed in range(2026, 2031)
        ]
        result = analyze_stability(
            predictions,
            metrics,
            ablations={
                "full": 0.05,
                "no_seat_stock": 0.04,
                "no_industry_stock": 0.04,
                "no_stock_stock": 0.04,
                "stock_only": 0.03,
            },
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["folds"]["0"]["top5_jaccard"], 1.0)
        self.assertEqual(result["regimes"]["up"]["sample_count"], 10.0)


if __name__ == "__main__":
    unittest.main()
