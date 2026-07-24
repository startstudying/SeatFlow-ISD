from __future__ import annotations

import unittest
from decimal import Decimal

from seatflow.backtest.trading_rules import price_limit_pct


class TradingRulesTest(unittest.TestCase):
    def test_price_limit_pct_by_a_share_board(self) -> None:
        fallback = Decimal("0.07")

        self.assertEqual(price_limit_pct("000001.SZ", fallback), Decimal("0.10"))
        self.assertEqual(price_limit_pct("600000.SH", fallback), Decimal("0.10"))
        self.assertEqual(price_limit_pct("300001.SZ", fallback), Decimal("0.20"))
        self.assertEqual(price_limit_pct("688001.SH", fallback), Decimal("0.20"))
        self.assertEqual(price_limit_pct("430001.BJ", fallback), Decimal("0.30"))
        self.assertEqual(price_limit_pct("200001.SZ", fallback), fallback)


if __name__ == "__main__":
    unittest.main()
