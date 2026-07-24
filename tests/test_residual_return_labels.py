from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

from seatflow.labels import (
    FIXTURE_ADJUSTMENT_POLICY,
    FIXTURE_CALENDAR_POLICY,
    StockLabel,
    build_candidate_pool_residual_labels,
)


class ResidualReturnLabelsTest(unittest.TestCase):
    def test_same_date_equal_weight_pool_is_subtracted(self) -> None:
        base = _label("000001.SZ", Decimal("0.03"))
        labels = (
            base,
            replace(base, target_stock_id="000002.SZ", future_return=Decimal("-0.01")),
        )
        residuals = build_candidate_pool_residual_labels(labels)

        self.assertEqual(residuals[0].candidate_pool_return, Decimal("0.01"))
        self.assertEqual(residuals[0].residual_return, Decimal("0.02"))
        self.assertEqual(residuals[1].residual_return, Decimal("-0.02"))
        self.assertEqual(sum(item.residual_return for item in residuals), Decimal("0"))

    def test_other_horizons_do_not_enter_one_day_pool(self) -> None:
        one_day = _label("000001.SZ", Decimal("0.03"))
        three_day = replace(one_day, horizon=3, future_return=Decimal("0.90"))

        residuals = build_candidate_pool_residual_labels((one_day, three_day))

        self.assertEqual(len(residuals), 1)
        self.assertEqual(residuals[0].residual_return, Decimal("0"))


def _label(stock_id: str, future_return: Decimal) -> StockLabel:
    as_of = date(2026, 1, 1)
    cutoff = datetime(2026, 1, 1, 18)
    return StockLabel(
        as_of_date=as_of,
        cutoff_time=cutoff,
        target_stock_id=stock_id,
        horizon=1,
        label_start_date=as_of + timedelta(days=1),
        label_end_date=as_of + timedelta(days=1),
        base_close=Decimal("10"),
        future_close=Decimal("10") * (Decimal("1") + future_return),
        future_return=future_return,
        direction_label=int(future_return > 0),
        label_available_time=cutoff + timedelta(days=1),
        adjustment_policy=FIXTURE_ADJUSTMENT_POLICY,
        calendar_policy=FIXTURE_CALENDAR_POLICY,
        base_source_name="fixture",
        future_source_name="fixture",
    )


if __name__ == "__main__":
    unittest.main()
