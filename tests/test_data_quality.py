from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.quality import validate_dragon_tiger_records
from seatflow.data.records import DragonTigerRecord, TradeSide


class DragonTigerRecordQualityTest(unittest.TestCase):
    def test_duplicate_canonical_record_is_an_error(self) -> None:
        record = DragonTigerRecord(
            trade_date=date(2026, 1, 15),
            seat_id="named-seat",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            event_reason="reason",
            source_name="test",
            source_row_count=2,
            source_side_set=("buy", "sell"),
        )

        report = validate_dragon_tiger_records((record, record))

        self.assertFalse(report.passed)
        self.assertTrue(any("duplicate canonical seat event" in error for error in report.errors))
        self.assertFalse(any("repeated seat event" in warning for warning in report.warnings))
