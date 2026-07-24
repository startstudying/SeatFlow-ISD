import unittest
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import cast

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.windows import DatasetManifest, build_window_samples
from seatflow.training.preflight import TimeSplit
from seatflow.training.splits import SplitPolicy, assign_fixed_time_split, reject_random_split


def _bar(trade_date: date, close: str) -> DailyBar:
    return DailyBar(
        trade_date=trade_date,
        stock_id="000001.SZ",
        open_price=Decimal(close),
        high_price=Decimal(close),
        low_price=Decimal(close),
        close_price=Decimal(close),
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="fixture",
        available_time=datetime.combine(trade_date, time(18, 0)),
    )


class TimeLeakageTest(unittest.TestCase):
    def test_window_samples_exclude_records_after_cutoff(self) -> None:
        as_of = date(2026, 1, 2)
        records = (
            DragonTigerRecord(
                trade_date=as_of,
                seat_id="early-seat",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=100.0,
                available_time=datetime(2026, 1, 2, 17, 0),
            ),
            DragonTigerRecord(
                trade_date=as_of,
                seat_id="late-seat",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=100.0,
                available_time=datetime(2026, 1, 2, 19, 0),
            ),
        )
        bars = tuple(_bar(date(2026, 1, day), str(10 + day)) for day in range(1, 4))

        samples = build_window_samples(
            records,
            bars,
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            cutoff_for_date=lambda _: datetime(2026, 1, 2, 18, 0),
            decay_lambda=0.1,
            max_edge_age_days=20,
        )

        self.assertEqual(len(samples), 1)
        edge_sources = {edge.source_id for edge in samples[0].graph.edges}
        self.assertIn("early-seat", edge_sources)
        self.assertNotIn("late-seat", edge_sources)
        self.assertEqual(samples[0].manifest()["target_stock_id"], "000001.SZ")

        manifest = DatasetManifest(
            samples=samples,
            split={"train": [samples[0].sample_id], "valid": [], "test": []},
            tensor_shapes={"seat": [1, 1]},
            missingness={"missing_available_time": 0},
        ).to_dict()
        self.assertIn("label_distribution", manifest)
        claims = cast(dict[str, bool], manifest["claims"])
        split_manifest = cast(dict[str, list[str]], manifest["split"])
        self.assertEqual(claims["data_ready"], False)
        self.assertEqual(split_manifest["train"], [samples[0].sample_id])

    def test_split_policy_rejects_short_embargo_and_random_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "embargo_days"):
            SplitPolicy(purge_days=1, embargo_days=2, max_horizon_days=5)
        with self.assertRaisesRegex(ValueError, "randomized"):
            reject_random_split("random")

    def test_fixed_split_purges_boundary_crossing_labels(self) -> None:
        as_of_dates = tuple(date(2026, 1, 1) + timedelta(days=offset) for offset in range(12))
        bars = tuple(_bar(day, str(10 + index)) for index, day in enumerate(as_of_dates))
        samples = build_window_samples(
            records=(),
            bars=bars,
            as_of_dates=as_of_dates[:8],
            target_stock_ids=("000001.SZ",),
            horizons=(2,),
            cutoff_for_date=lambda value: datetime.combine(value, time(18, 0)),
            decay_lambda=0.1,
            max_edge_age_days=20,
        )
        split = TimeSplit(
            train_start=date(2026, 1, 1),
            train_end=date(2026, 1, 4),
            valid_start=date(2026, 1, 5),
            valid_end=date(2026, 1, 8),
            test_start=date(2026, 1, 9),
            test_end=date(2026, 1, 12),
        )

        assignments = assign_fixed_time_split(
            samples,
            split,
            SplitPolicy(purge_days=1, embargo_days=2, max_horizon_days=2),
        )

        self.assertNotIn("2026-01-03::000001.SZ", assignments["train"])
        self.assertNotIn("2026-01-05::000001.SZ", assignments["valid"])


if __name__ == "__main__":
    unittest.main()
