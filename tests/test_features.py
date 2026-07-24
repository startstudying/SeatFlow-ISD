import unittest
from datetime import date
from decimal import Decimal

from seatflow.data.contracts import DailyBar
from seatflow.features import compute_basic_features


class FeatureTest(unittest.TestCase):
    def test_features_use_trailing_observations_only(self) -> None:
        bars = tuple(
            DailyBar(
                trade_date=date(2026, 1, day),
                stock_id="000001.SZ",
                open_price=Decimal("10"),
                high_price=Decimal(str(10 + day)),
                low_price=Decimal("9"),
                close_price=Decimal(str(10 + day)),
                volume=Decimal("100"),
                amount=Decimal("1000"),
                source_name="test",
            )
            for day in range(1, 6)
        )

        features = compute_basic_features(bars)

        self.assertIsNone(features[3].ma_5)
        self.assertEqual(features[4].ma_5, Decimal("13"))
        self.assertEqual(features[4].return_1d, Decimal(1) / Decimal(14))
