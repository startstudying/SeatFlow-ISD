from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import (
    INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    DragonTigerRecord,
    TradeSide,
)
from seatflow.r4.decayed_seat_state import DecayedSeatState
from seatflow.r4_features import (
    MISSING_SEAT_EVENT_FEATURE,
    SEAT_AGGREGATE_FEATURES,
    build_stock_seat_aggregate_features_asof,
)


class SeatAggregateAsOfTest(unittest.TestCase):
    def test_current_day_aggregate_features_are_stock_level_and_pit(self) -> None:
        as_of = date(2026, 1, 5)
        rows = build_stock_seat_aggregate_features_asof(
            records=(
                _record(date(2026, 1, 3), "seat-repeat", "000001.SZ", TradeSide.BUY, 50.0),
                _record(as_of, "seat-repeat", "000001.SZ", TradeSide.BUY, 100.0),
                _record(
                    as_of,
                    "institution-proxy",
                    "000001.SZ",
                    TradeSide.BUY,
                    300.0,
                    identity_scope=INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
                ),
                _record(
                    as_of,
                    "沪股通专用",
                    "000001.SZ",
                    TradeSide.SELL,
                    200.0,
                ),
                _record(as_of, "future-visible-only", "000002.SZ", TradeSide.BUY, 999.0),
            ),
            as_of_date=as_of,
            stock_ids=("000001.SZ",),
            decayed_states=(
                DecayedSeatState(
                    trade_date=as_of.isoformat(),
                    stock_id="000001.SZ",
                    last_seat_event_date=as_of.isoformat(),
                    days_since_last_seat_event=0,
                    last_net_buy_amount=200.0,
                    last_seat_count=3,
                    last_institution_ratio=1.0 / 3.0,
                    last_seat_concentration=14.0 / 36.0,
                    last_seat_signal=8.0,
                    decayed_seat_signal=8.0,
                    has_new_seat_event=True,
                ),
            ),
        )

        row = next(item for item in rows if item.stock_id == "000001.SZ")
        self.assertEqual(tuple(row.features), SEAT_AGGREGATE_FEATURES)
        self.assertEqual(row.features["seat_buy_amount"], 400.0)
        self.assertEqual(row.features["seat_sell_amount"], 200.0)
        self.assertEqual(row.features["seat_net_amount"], 200.0)
        self.assertEqual(row.features["seat_count"], 3.0)
        self.assertAlmostEqual(row.features["institution_seat_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(row.features["northbound_seat_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(row.features["seat_concentration"], 14.0 / 36.0)
        self.assertAlmostEqual(row.features["seat_repeat_participation"], 1.0 / 3.0)
        self.assertEqual(row.features["days_since_last_event"], 0.0)
        self.assertEqual(row.features["decayed_seat_signal"], 8.0)
        self.assertFalse(row.missing_seat_event)
        self.assertEqual(row.model_features()[MISSING_SEAT_EVENT_FEATURE], 0.0)

    def test_missing_current_event_keeps_explicit_marker_and_decayed_signal(self) -> None:
        as_of = date(2026, 1, 5)
        rows = build_stock_seat_aggregate_features_asof(
            records=(_record(date(2026, 1, 3), "seat-old", "000001.SZ", TradeSide.BUY, 50.0),),
            as_of_date=as_of,
            stock_ids=("000001.SZ",),
            decayed_states=(
                DecayedSeatState(
                    trade_date=as_of.isoformat(),
                    stock_id="000001.SZ",
                    last_seat_event_date="2026-01-03",
                    days_since_last_seat_event=2,
                    last_net_buy_amount=50.0,
                    last_seat_count=1,
                    last_institution_ratio=0.0,
                    last_seat_concentration=1.0,
                    last_seat_signal=4.0,
                    decayed_seat_signal=2.0,
                    has_new_seat_event=False,
                ),
            ),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row.missing_seat_event)
        self.assertEqual(row.model_features()[MISSING_SEAT_EVENT_FEATURE], 1.0)
        self.assertEqual(row.features["seat_buy_amount"], 0.0)
        self.assertEqual(row.features["days_since_last_event"], 2.0)
        self.assertEqual(row.features["decayed_seat_signal"], 2.0)


def _record(
    trade_date: date,
    seat_id: str,
    stock_id: str,
    side: TradeSide,
    amount: float,
    *,
    identity_scope: str = "named_seat",
) -> DragonTigerRecord:
    return DragonTigerRecord(
        trade_date=trade_date,
        seat_id=seat_id,
        stock_id=stock_id,
        side=side,
        amount=amount,
        identity_scope=identity_scope,
    )


if __name__ == "__main__":
    unittest.main()
