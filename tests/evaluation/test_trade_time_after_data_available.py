from __future__ import annotations

import unittest
from datetime import UTC, datetime

from seatflow.evaluation.time_alignment import (
    TradeTimeAlignmentAudit,
    assert_trade_time_after_data_available,
    evaluate_trade_time_alignment,
)


class TradeTimeAlignmentTest(unittest.TestCase):
    def test_assert_passes_when_trade_time_is_not_before_available_time(self) -> None:
        assert_trade_time_after_data_available(
            datetime(2026, 1, 2, 18, 0, tzinfo=UTC),
            datetime(2026, 1, 2, 17, 0, tzinfo=UTC),
            context="test-row",
        )

    def test_assert_raises_when_trade_time_precedes_data_available_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot precede available_time"):
            assert_trade_time_after_data_available(
                datetime(2026, 1, 2, 17, 0),
                datetime(2026, 1, 2, 18, 0),
                context="bad-row",
            )

    def test_evaluate_returns_insufficient_when_records_do_not_expose_times(self) -> None:
        result = evaluate_trade_time_alignment(
            (
                {"trade_time": datetime(2026, 1, 2, 18, 0), "data_available_time": None},
                {
                    "trade_time": datetime(2026, 1, 3, 18, 0),
                    "data_available_time": datetime(2026, 1, 3, 18, 0),
                },
            )
        )
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.checked_records, 1)
        self.assertIn("missing", result.evidence_gaps[0])

    def test_evaluate_confirmed_for_valid_records(self) -> None:
        result = evaluate_trade_time_alignment(
            (
                {
                    "trade_time": datetime(2026, 1, 2, 18, 0),
                    "data_available_time": datetime(2026, 1, 2, 17, 0),
                },
                {
                    "trade_time": datetime(2026, 1, 3, 18, 0),
                    "data_available_time": datetime(2026, 1, 3, 17, 0),
                },
            )
        )
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.checked_records, 2)
        self.assertEqual(result.evidence_gaps, ())

    def test_no_records_is_not_an_error(self) -> None:
        result = evaluate_trade_time_alignment(())
        self.assertIsInstance(result, TradeTimeAlignmentAudit)
        self.assertEqual(result.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
