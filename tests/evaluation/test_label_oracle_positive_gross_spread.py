from __future__ import annotations

import unittest

from seatflow.evaluation.oracle_audit import label_oracle


class LabelOraclePositiveGrossSpreadTest(unittest.TestCase):
    def test_label_oracle_marks_top5_gross_spread_positive(self) -> None:
        rows = [
            {
                "as_of_date": "2026-01-01",
                "stock_id": "A",
                "score": 1.0,
                "label_return": 0.02,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "B",
                "score": 0.9,
                "label_return": 0.015,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "C",
                "score": 0.8,
                "label_return": 0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "D",
                "score": 0.1,
                "label_return": -0.02,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "E",
                "score": -0.1,
                "label_return": -0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "F",
                "score": -0.2,
                "label_return": -0.01,
            },
        ]
        result = label_oracle(rows, top_k=5)

        self.assertEqual(result.top_k, 5)
        self.assertGreater(result.gross_spread, 0.0)


if __name__ == "__main__":
    unittest.main()

