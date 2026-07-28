"""Phase 2 tests: PIT industry membership wiring for verified snapshots.

The sw2021 L1 industry membership backfill is sitting on disk but, until
this wiring, the ``industry`` field on every seat event was empty so the
graph builder had nothing to attach ``belongs_to`` edges to. These tests
prove the load path fills the field point-in-time without changing the
default behaviour for callers that omit ``industry_membership_root``.
"""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.data.store import SnapshotStore
from seatflow.datasets.snapshots import (
    load_verified_historical_snapshots,
)


def _cutoff(trade_date: date) -> datetime:
    return datetime.combine(trade_date, time(18, 0))


def _write_snapshots(root: Path) -> tuple[date, ...]:
    store = SnapshotStore(root / "processed")
    calendar = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(5))
    metadata = {"quality": {"passed": True}}
    for index, trade_date in enumerate(calendar):
        bar = DailyBar(
            trade_date=trade_date,
            stock_id="000001.SZ",
            open_price=Decimal("10") + Decimal(index),
            high_price=Decimal("11") + Decimal(index),
            low_price=Decimal("9") + Decimal(index),
            close_price=Decimal("10") + Decimal(index),
            volume=Decimal("100000"),
            amount=Decimal("1000000"),
            source_name="tushare",
            source_version="fixture-v1",
            available_time=_cutoff(trade_date),
            volume_unit="shares",
            amount_unit="cny",
        )
        event = DragonTigerRecord(
            trade_date=trade_date,
            seat_id="institutional_proxy",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=1_000_000.0,
            available_time=_cutoff(trade_date),
            source_name="tushare",
            source_version="top_inst",
        )
        store.write("tushare", "daily_bar", trade_date, (bar,), metadata=metadata)
        store.write(
            "tushare",
            "dragon_tiger_seat_event",
            trade_date,
            (event,),
            metadata=metadata,
        )
    return calendar


def _write_industry_membership(
    root: Path,
    *,
    rows: list[dict[str, object]],
    l1_code: str = "801010.SI",
) -> None:
    membership_dir = (
        root
        / "r1_industry_membership"
        / "tushare"
        / "sw2021_l1_industry_membership"
        / f"l1_code={l1_code}"
        / "is_new=Y"
    )
    membership_dir.mkdir(parents=True, exist_ok=True)
    (membership_dir / "data.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class IndustryWiringTest(unittest.TestCase):
    def test_default_loader_leaves_industry_unset(self) -> None:
        """Without industry_membership_root the loader must not invent data;
        it is purely an opt-in wiring so R5 evidence stays reproducible."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            snapshots = load_verified_historical_snapshots(
                root, provider="tushare", trading_calendar=calendar
            )
            self.assertTrue(snapshots.seat_events)
            for event in snapshots.seat_events:
                self.assertIsNone(event.industry)

    def test_fills_industry_point_in_time(self) -> None:
        """A stock's industry on a given trade_date is its L1 membership
        with in_date <= trade_date < out_date. The wiring should pick up
        the in_date from the backfill and stop applying on out_date."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            # 000001.SZ is a member from 2026-01-02 to 2026-01-04 inclusive.
            # 2026-01-01 is before in_date, so day 0 keeps industry=None.
            _write_industry_membership(
                root,
                rows=[
                    {
                        "stock_id": "000001.SZ",
                        "l1_code": "801010.SI",
                        "in_date": "2026-01-02",
                        "out_date": "2026-01-05",
                        "is_new": "Y",
                    }
                ],
            )
            snapshots = load_verified_historical_snapshots(
                root,
                provider="tushare",
                trading_calendar=calendar,
                industry_membership_root=root / "r1_industry_membership",
            )
            industries = {
                event.trade_date: event.industry for event in snapshots.seat_events
            }
            self.assertIsNone(industries[calendar[0]])  # 2026-01-01
            self.assertEqual(industries[calendar[1]], "801010.SI")  # 2026-01-02
            self.assertEqual(industries[calendar[2]], "801010.SI")  # 2026-01-03
            self.assertEqual(industries[calendar[3]], "801010.SI")  # 2026-01-04
            self.assertIsNone(industries[calendar[4]])  # 2026-01-05 (out_date)

    def test_does_not_overwrite_existing_industry(self) -> None:
        """If the seat-event row already carries an industry (e.g. a
        verified backfill that pre-populated it), the loader must not
        silently overwrite it with the membership lookup."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            _write_industry_membership(
                root,
                rows=[
                    {
                        "stock_id": "000001.SZ",
                        "l1_code": "801010.SI",
                        "in_date": "2026-01-01",
                        "out_date": None,
                        "is_new": "Y",
                    }
                ],
            )
            # Rewrite the first day's seat-event snapshot with an explicit
            # industry so the wiring must respect it instead of overwriting.
            store = SnapshotStore(root / "processed")
            trade_date = calendar[0]
            explicit_event = DragonTigerRecord(
                trade_date=trade_date,
                seat_id="institutional_proxy",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=1_000_000.0,
                industry="999999.SI",
                available_time=_cutoff(trade_date),
                source_name="tushare",
                source_version="top_inst",
            )
            store.write(
                "tushare",
                "dragon_tiger_seat_event",
                trade_date,
                (explicit_event,),
                metadata={"quality": {"passed": True}},
            )
            snapshots = load_verified_historical_snapshots(
                root,
                provider="tushare",
                trading_calendar=calendar,
                industry_membership_root=root / "r1_industry_membership",
            )
            first_event = next(
                event for event in snapshots.seat_events
                if event.trade_date == trade_date
            )
            self.assertEqual(first_event.industry, "999999.SI")

    def test_load_industry_membership_map_resolves_overrides(self) -> None:
        """A stock that switches industry mid-window must use the most
        recent L1 code, not the first one. This is what `_fill_industry`
        threads through to the loader's tracked dict."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_industry_membership(
                root,
                rows=[
                    {
                        "stock_id": "000001.SZ",
                        "l1_code": "801010.SI",
                        "in_date": "2026-01-01",
                        "out_date": "2026-01-03",
                        "is_new": "Y",
                    }
                ],
                l1_code="801010.SI",
            )
            _write_industry_membership(
                root,
                rows=[
                    {
                        "stock_id": "000001.SZ",
                        "l1_code": "801020.SI",
                        "in_date": "2026-01-03",
                        "out_date": None,
                        "is_new": "Y",
                    }
                ],
                l1_code="801020.SI",
            )
            calendar = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(3))
            for trade_date in calendar:
                bar = DailyBar(
                    trade_date=trade_date,
                    stock_id="000001.SZ",
                    open_price=Decimal("10"),
                    high_price=Decimal("11"),
                    low_price=Decimal("9"),
                    close_price=Decimal("10"),
                    volume=Decimal("100000"),
                    amount=Decimal("1000000"),
                    source_name="tushare",
                    source_version="fixture-v1",
                    available_time=_cutoff(trade_date),
                    volume_unit="shares",
                    amount_unit="cny",
                )
                event = DragonTigerRecord(
                    trade_date=trade_date,
                    seat_id="institutional_proxy",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=1_000_000.0,
                    available_time=_cutoff(trade_date),
                    source_name="tushare",
                    source_version="top_inst",
                )
                metadata = {"quality": {"passed": True}}
                store = SnapshotStore(root / "processed")
                store.write("tushare", "daily_bar", trade_date, (bar,), metadata=metadata)
                store.write(
                    "tushare",
                    "dragon_tiger_seat_event",
                    trade_date,
                    (event,),
                    metadata=metadata,
                )

            snapshots = load_verified_historical_snapshots(
                root,
                provider="tushare",
                trading_calendar=calendar,
                industry_membership_root=root / "r1_industry_membership",
            )
            industries = {
                event.trade_date: event.industry for event in snapshots.seat_events
            }
            self.assertEqual(industries[calendar[0]], "801010.SI")
            self.assertEqual(industries[calendar[1]], "801010.SI")
            self.assertEqual(industries[calendar[2]], "801020.SI")

    def test_missing_industry_root_raises(self) -> None:
        """The wiring is opt-in but, when opted in, the root must exist;
        silently swallowing a typo would re-create the original bug."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            with self.assertRaisesRegex(ValueError, "industry membership root does not exist"):
                load_verified_historical_snapshots(
                    root,
                    provider="tushare",
                    trading_calendar=calendar,
                    industry_membership_root=root / "no_such_directory",
                )


if __name__ == "__main__":
    unittest.main()
