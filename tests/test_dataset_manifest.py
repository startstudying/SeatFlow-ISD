import unittest
from datetime import date, datetime, time
from decimal import Decimal
from typing import cast

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.windows import DatasetManifest, build_window_samples
from seatflow.labels import FROZEN_ADJUSTMENT_POLICY, FROZEN_CALENDAR_POLICY


def _bar(day: int, close: str) -> DailyBar:
    trade_date = date(2026, 1, day)
    return DailyBar(
        trade_date=trade_date,
        stock_id="000001.SZ",
        open_price=Decimal(close),
        high_price=Decimal(close),
        low_price=Decimal(close),
        close_price=Decimal(close),
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="unit-test",
        available_time=datetime.combine(trade_date, time(18, 0)),
        source_version="2026-01-fixture",
        volume_unit="share",
        amount_unit="cny",
    )


class DatasetManifestTest(unittest.TestCase):
    def test_graph_stock_universe_filters_unrelated_window_records(self) -> None:
        as_of = date(2026, 1, 2)
        records = tuple(
            DragonTigerRecord(
                trade_date=as_of,
                seat_id=f"seat-{stock_id}",
                stock_id=stock_id,
                side=TradeSide.BUY,
                amount=100.0,
                available_time=datetime(2026, 1, 2, 18, 0),
            )
            for stock_id in ("000001.SZ", "000002.SZ")
        )
        samples = build_window_samples(
            records=records,
            bars=tuple(_bar(day, str(10 + day)) for day in range(1, 8)),
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            cutoff_for_date=lambda _: datetime(2026, 1, 2, 18, 0),
            decay_lambda=0.1,
            max_edge_age_days=20,
            graph_stock_ids_by_date={as_of: ("000001.SZ",)},
        )

        stock_ids = {
            node.node_id for node in samples[0].graph.nodes if node.node_type.value == "stock"
        }
        self.assertEqual(stock_ids, {"000001.SZ"})

    def test_manifest_carries_real_research_label_contract_fields(self) -> None:
        bars = tuple(_bar(day, str(10 + day)) for day in range(1, 8))
        samples = build_window_samples(
            records=(),
            bars=bars,
            as_of_dates=(date(2026, 1, 2),),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: datetime(2026, 1, 2, 18, 0),
            decay_lambda=0.1,
            max_edge_age_days=20,
            adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
            calendar_policy=FROZEN_CALENDAR_POLICY,
            trading_calendar=tuple(bar.trade_date for bar in bars),
        )

        manifest = DatasetManifest(
            samples=samples,
            split={"train": [samples[0].sample_id], "valid": [], "test": []},
            tensor_shapes={"seat": [0, 0]},
            missingness={"insufficient_future_calendar": 0},
        ).to_dict()

        self.assertEqual(manifest["horizon_distribution"], {"1": 1, "3": 1, "5": 1})
        self.assertRegex(cast(str, manifest["source_hash"]), r"^[0-9a-f]{64}$")
        self.assertIn("feature_schema", manifest)
        self.assertIn("label_schema", manifest)
        self.assertIn("missing_reasons", manifest)

        sample = cast(list[dict[str, object]], manifest["samples"])[0]
        self.assertEqual(sample["as_of_date"], "2026-01-02")
        self.assertEqual(sample["cutoff_time"], "2026-01-02T18:00:00")
        windows = cast(list[dict[str, object]], sample["label_windows"])
        self.assertEqual([window["horizon"] for window in windows], [1, 3, 5])
        self.assertEqual(windows[0]["start"], "2026-01-03")
        self.assertEqual(windows[0]["direction_label"], 1)
        self.assertEqual(windows[0]["future_return"], str(Decimal("1") / Decimal("12")))
        readiness = cast(dict[str, object], manifest["data_readiness"])
        claims = cast(dict[str, bool], manifest["claims"])
        self.assertTrue(readiness["real_research_ready"])
        self.assertTrue(readiness["frozen_1_3_5_policy"])
        self.assertTrue(readiness["source_hashes_verified"])
        self.assertTrue(claims["data_ready"])

    def test_manifest_fails_closed_for_fixture_label_policy(self) -> None:
        bars = tuple(_bar(day, str(10 + day)) for day in range(1, 8))
        samples = build_window_samples(
            records=(),
            bars=bars,
            as_of_dates=(date(2026, 1, 2),),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: datetime(2026, 1, 2, 18, 0),
            decay_lambda=0.1,
            max_edge_age_days=20,
        )

        manifest = DatasetManifest(
            samples=samples,
            split={"train": [samples[0].sample_id], "valid": [], "test": []},
            tensor_shapes={"seat": [0, 0]},
            missingness={"insufficient_future_calendar": 0},
        ).to_dict()

        readiness = cast(dict[str, object], manifest["data_readiness"])
        claims = cast(dict[str, bool], manifest["claims"])
        self.assertFalse(readiness["real_research_ready"])
        self.assertFalse(readiness["frozen_1_3_5_policy"])
        self.assertFalse(claims["data_ready"])

    def test_build_window_samples_passes_bars_to_stock_correlation_edges(self) -> None:
        as_of = date(2026, 1, 13)
        cutoff = datetime(2026, 1, 13, 18, 0)
        samples = build_window_samples(
            records=(),
            bars=_bars_for_correlation_and_labels(),
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=0.1,
            max_edge_age_days=20,
            adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
            calendar_policy=FROZEN_CALENDAR_POLICY,
            trading_calendar=tuple(date(2026, 1, day) for day in range(9, 19)),
        )

        relations = {edge.relation for edge in samples[0].graph.edges}
        self.assertIn("positively_correlated_with", relations)

    def test_manifest_fails_closed_for_data_readiness_claim(self) -> None:
        manifest = DatasetManifest(
            samples=(),
            split={"train": [], "valid": [], "test": []},
            tensor_shapes={},
            missingness={"raw_close_fixture_only": 1},
        ).to_dict()

        readiness = cast(dict[str, object], manifest["data_readiness"])
        claims = cast(dict[str, bool], manifest["claims"])
        self.assertFalse(readiness["real_research_ready"])
        self.assertFalse(claims["data_ready"])
def _bars_for_correlation_and_labels() -> tuple[DailyBar, ...]:
    closes = {
        "000001.SZ": tuple(Decimal(10 + offset) for offset in range(10)),
        "000002.SZ": tuple(Decimal(20 + 2 * offset) for offset in range(10)),
    }
    days = tuple(date(2026, 1, day) for day in range(9, 19))
    bars: list[DailyBar] = []
    for stock_id, values in closes.items():
        for day, close in zip(days, values, strict=True):
            bars.append(_stock_bar(stock_id, day, close))
    return tuple(bars)


def _stock_bar(stock_id: str, trade_date: date, close: Decimal) -> DailyBar:
    return DailyBar(
        trade_date=trade_date,
        stock_id=stock_id,
        open_price=close,
        high_price=close,
        low_price=close,
        close_price=close,
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="unit-test",
        available_time=datetime.combine(trade_date, time(18, 0)),
        source_version="2026-01-fixture",
        volume_unit="share",
        amount_unit="cny",
    )


if __name__ == "__main__":
    unittest.main()
