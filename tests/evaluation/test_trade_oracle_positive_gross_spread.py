from __future__ import annotations

import unittest

from seatflow.evaluation.oracle_audit import trade_return_oracle


class TradeOraclePositiveGrossSpreadTest(unittest.TestCase):
    def test_trade_return_oracle_marks_top5_gross_spread_positive(self) -> None:
        rows = [
            {
                "as_of_date": "2026-01-01",
                "stock_id": "AAA",
                "score": 0.5,
                "trade_return": 0.03,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "BBB",
                "score": 0.4,
                "trade_return": 0.02,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "CCC",
                "score": 0.3,
                "trade_return": 0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "DDD",
                "score": 0.2,
                "trade_return": 0.00,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "EEE",
                "score": 0.1,
                "trade_return": -0.01,
            },
            {
                "as_of_date": "2026-01-01",
                "stock_id": "FFF",
                "score": 0.0,
                "trade_return": -0.02,
            },
        ]
        result = trade_return_oracle(rows, top_k=5)

        self.assertEqual(result.top_k, 5)
        self.assertGreater(result.gross_spread, 0.0)


if __name__ == "__main__":
    unittest.main()

