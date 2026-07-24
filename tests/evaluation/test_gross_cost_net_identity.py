from __future__ import annotations

import unittest

from seatflow.evaluation.cost_attribution import gross_cost_net_identity


class GrossCostNetIdentityTest(unittest.TestCase):
    def test_candidate_baseline_gross_cost_net_identity_holds(self) -> None:
        candidate_rows = [
            {
                "as_of_date": "2026-01-01",
                "stock_id": "A",
                "selection_score": 1.0,
                "label_return": 0.02,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "B",
                "selection_score": 0.9,
                "label_return": 0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "C",
                "selection_score": 0.0,
                "label_return": -0.02,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "A",
                "selection_score": 1.1,
                "label_return": 0.01,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "B",
                "selection_score": 0.8,
                "label_return": 0.00,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "C",
                "selection_score": 0.1,
                "label_return": -0.01,
            },
        ]
        baseline_rows = [
            {
                "as_of_date": "2026-01-01",
                "stock_id": "A",
                "selection_score": 0.7,
                "label_return": 0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "B",
                "selection_score": 0.6,
                "label_return": 0.00,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "C",
                "selection_score": 0.2,
                "label_return": -0.01,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "A",
                "selection_score": 0.9,
                "label_return": 0.02,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "B",
                "selection_score": 0.5,
                "label_return": 0.01,
            },
            {
                "as_of_date": "2026-01-02",
                "stock_id": "C",
                "selection_score": 0.1,
                "label_return": -0.02,
            },
        ]
        report = gross_cost_net_identity(candidate_rows, baseline_rows, top_k=2)

        self.assertAlmostEqual(
            report.candidate_net_spread,
            report.candidate_gross_spread - report.candidate_total_cost,
        )
        self.assertAlmostEqual(
            report.baseline_net_spread,
            report.baseline_gross_spread - report.baseline_total_cost,
        )
        self.assertAlmostEqual(
            report.spread_net_from_identity,
            report.candidate_gross_spread
            - report.baseline_gross_spread
            - report.candidate_total_cost
            + report.baseline_total_cost,
        )


if __name__ == "__main__":
    unittest.main()

