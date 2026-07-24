from __future__ import annotations

import unittest
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.data.store import SnapshotStore
from seatflow.datasets.snapshots import (
    MaturedLabelDiskCache,
    build_real_window_samples,
    load_verified_historical_snapshots,
)


class VerifiedSnapshotWindowsTest(unittest.TestCase):
    def test_matured_label_cache_round_trips_without_memory_only_state(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache = MaturedLabelDiskCache(Path(temporary_directory) / "labels.sqlite3")
            key = (date(2024, 1, 2), "000001.SZ")
            cache[key] = ()
            self.assertEqual(cache[key], ())
            cache.close()

    def test_loads_manifest_verified_snapshots_and_blocks_unreleased_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            snapshots = load_verified_historical_snapshots(
                root,
                provider="tushare",
                trading_calendar=calendar,
            )

            self.assertEqual(len(snapshots.trading_calendar), len(calendar))
            self.assertEqual(len(snapshots.manifest_hashes), len(calendar) * 2)
            self.assertEqual(len(snapshots.source_hash), 64)

            with self.assertRaisesRegex(ValueError, "no fully observed"):
                build_real_window_samples(
                    snapshots,
                    target_stock_ids_by_date={calendar[1]: ("000001.SZ",)},
                    cutoff_for_date=_cutoff,
                    observed_label_end=calendar[5],
                    decay_lambda=0.1,
                    max_edge_age_days=20,
                )

            samples = build_real_window_samples(
                snapshots,
                target_stock_ids_by_date={calendar[1]: ("000001.SZ",)},
                cutoff_for_date=_cutoff,
                observed_label_end=calendar[6],
                decay_lambda=0.1,
                max_edge_age_days=20,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(tuple(label.horizon for label in samples[0].labels), (1, 3, 5))
            self.assertTrue(all(label.label_end_date <= calendar[6] for label in samples[0].labels))
            seat = next(
                node for node in samples[0].graph.nodes if node.node_type.value == "seat"
            )
            self.assertEqual(seat.features["seat_profile_available"], 1.0)
            self.assertGreater(seat.features["seat_profile_event_count"], 0.0)

    def test_rejects_tampered_snapshot_before_parsing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = _write_snapshots(root)
            data_path = (
                root
                / "processed"
                / "tushare"
                / "daily_bar"
                / f"date={calendar[0].isoformat()}"
                / "data.json"
            )
            data_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest verification failed"):
                load_verified_historical_snapshots(
                    root,
                    provider="tushare",
                    trading_calendar=calendar,
                )


def _write_snapshots(root: Path) -> tuple[date, ...]:
    store = SnapshotStore(root / "processed")
    calendar = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(9))
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


def _cutoff(trade_date: date) -> datetime:
    return datetime.combine(trade_date, time(18, 0))


if __name__ == "__main__":
    unittest.main()
