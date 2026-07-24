import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from math import log1p
from typing import cast

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import (
    EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
    tensorize_window_samples,
)
from seatflow.datasets.windows import build_window_samples
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import (
    TENSORIZATION_VERSION,
    FrozenGraphSchema,
    PyGUnavailableError,
    derive_frozen_graph_schema,
    tensorize_hetero_graph,
    tensorize_hetero_graphs,
)
from seatflow.graph.types import NodeType
from seatflow.labels import (
    FROZEN_ADJUSTMENT_POLICY,
    FROZEN_CALENDAR_POLICY,
    LABEL_V2_ADJUSTMENT_POLICY,
    LABEL_V2_PRICE_POLICY,
    StockLabelV2,
)


class PyGTensorizeTest(unittest.TestCase):
    def test_tensorizes_all_node_types_with_stable_mapping_or_clear_skip(self) -> None:
        graph = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
            (
                DragonTigerRecord(
                    trade_date=date(2026, 1, 2),
                    seat_id="seat-b",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    industry="banking",
                    concepts=("large_cap",),
                    event_reason="daily_deviation",
                ),
                DragonTigerRecord(
                    trade_date=date(2026, 1, 2),
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.SELL,
                    amount=50.0,
                    industry="banking",
                    concepts=("large_cap",),
                    event_reason="daily_deviation",
                ),
            ),
            as_of_date=date(2026, 1, 2),
            bars=(
                _daily_bar("000001.SZ", date(2026, 1, 1), 10),
                _daily_bar("000001.SZ", date(2026, 1, 2), 11),
            ),
        )

        try:
            tensorized = tensorize_hetero_graph(graph)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        metadata = tensorized.metadata
        node_mapping = cast(dict[str, dict[str, int]], metadata["node_mapping"])
        edge_shapes = cast(dict[str, list[int]], metadata["edge_shapes"])
        edge_attr_schema = cast(dict[str, list[str]], metadata["edge_attr_schema"])
        feature_schema = cast(dict[str, list[str]], metadata["feature_schema"])
        coverage_audit = cast(dict[str, object], metadata["coverage_audit"])
        self.assertEqual(set(node_mapping), {node_type.value for node_type in NodeType})
        self.assertEqual(metadata["tensorization_version"], TENSORIZATION_VERSION)
        self.assertEqual(node_mapping["seat"], {"seat-a": 0, "seat-b": 1})
        self.assertIn("stock_bar_count", feature_schema["stock"])
        bar_count_index = feature_schema["stock"].index("stock_bar_count")
        self.assertAlmostEqual(
            float(tensorized.data["stock"].x[0, bar_count_index]), log1p(2.0), places=5
        )
        self.assertIn("seat::buys::stock", edge_shapes)
        self.assertIn("stock::belongs_to_industry::industry", edge_shapes)
        buys = tensorized.data[("seat", "buys", "stock")]
        self.assertTrue(hasattr(buys, "edge_attr"))
        self.assertIn("edge_gate_input", edge_attr_schema["seat::buys::stock"])
        gate_index = edge_attr_schema["seat::buys::stock"].index("edge_gate_input")
        self.assertEqual(float(buys.edge_attr[0, gate_index]), float(buys.edge_weight[0]))
        amount_index = edge_attr_schema["seat::buys::stock"].index("amount")
        self.assertAlmostEqual(float(buys.edge_attr[0, amount_index]), log1p(100.0), places=5)
        self.assertEqual(coverage_audit["industry_node_count"], 1)
        self.assertEqual(coverage_audit["concept_node_count"], 1)
        self.assertEqual(coverage_audit["missing_static_node_types"], [])

    def test_metadata_reports_missing_industry_and_concept_coverage(self) -> None:
        as_of_date = date(2026, 1, 2)
        graph = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
            (
                DragonTigerRecord(
                    trade_date=as_of_date,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                ),
            ),
            as_of_date,
        )

        try:
            tensorized = tensorize_hetero_graph(graph)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        coverage_audit = cast(dict[str, object], tensorized.metadata["coverage_audit"])
        self.assertEqual(coverage_audit["industry_node_count"], 0)
        self.assertEqual(coverage_audit["concept_node_count"], 0)
        self.assertEqual(
            coverage_audit["missing_static_node_types"], ["industry", "concept"]
        )

    def test_frozen_schema_materializes_missing_edge_types_with_shared_metadata(
        self,
    ) -> None:
        builder = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20)
        as_of_date = date(2026, 1, 13)
        record = DragonTigerRecord(
            trade_date=as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
        )
        graph_with_correlation = builder.build(
            (record,),
            as_of_date,
            bars=_bars_for_correlation(),
        )
        graph_without_correlation = builder.build(
            (record,),
            as_of_date,
            bars=_bars_for_correlation()[:2],
        )

        try:
            with_edges, without_edges = tensorize_hetero_graphs(
                (graph_with_correlation, graph_without_correlation)
            )
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        correlation_type = ("stock", "positively_correlated_with", "stock")
        self.assertEqual(with_edges.data.metadata(), without_edges.data.metadata())
        self.assertIn(correlation_type, without_edges.data.edge_types)
        self.assertGreater(int(with_edges.data[correlation_type].edge_index.size(1)), 0)
        self.assertEqual(int(without_edges.data[correlation_type].edge_index.size(1)), 0)
        self.assertEqual(int(without_edges.data[correlation_type].edge_attr.size(0)), 0)
        self.assertEqual(int(without_edges.data[correlation_type].edge_weight.numel()), 0)

    def test_frozen_schema_conflict_fails_closed(self) -> None:
        as_of_date = date(2026, 1, 13)
        graph = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
            (
                DragonTigerRecord(
                    trade_date=as_of_date,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                ),
            ),
            as_of_date,
        )
        schema = derive_frozen_graph_schema((graph,))
        edge_type = schema.edge_types[0]
        broken_schema = FrozenGraphSchema(
            node_types=schema.node_types,
            edge_types=schema.edge_types,
            node_feature_schema=schema.node_feature_schema,
            edge_attr_schema={edge_type: ("edge_gate_input", "graph_edge_weight")},
        )

        try:
            with self.assertRaisesRegex(ValueError, "schema conflict"):
                tensorize_hetero_graph(graph, frozen_schema=broken_schema)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

    def test_bulk_window_tensorize_fails_closed_for_fixture_label_policy(self) -> None:
        as_of = date(2026, 1, 2)
        cutoff = datetime(2026, 1, 2, 18, 0)
        samples = build_window_samples(
            records=(
                DragonTigerRecord(
                    trade_date=as_of,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    available_time=cutoff - timedelta(hours=1),
                ),
            ),
            bars=tuple(
                _daily_bar("000001.SZ", as_of + timedelta(days=offset), 10 + offset)
                for offset in range(6)
            ),
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=0.1,
            max_edge_age_days=20,
        )

        with self.assertRaisesRegex(ValueError, "frozen adjustment policy"):
            tensorize_window_samples(samples)

    def test_bulk_window_tensorize_accepts_frozen_policy(self) -> None:
        as_of = date(2026, 1, 2)
        cutoff = datetime(2026, 1, 2, 18, 0)
        bars = tuple(
            _daily_bar("000001.SZ", as_of + timedelta(days=offset), 10 + offset)
            for offset in range(6)
        )
        samples = build_window_samples(
            records=(
                DragonTigerRecord(
                    trade_date=as_of,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    available_time=cutoff - timedelta(hours=1),
                ),
            ),
            bars=bars,
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=0.1,
            max_edge_age_days=20,
            adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
            calendar_policy=FROZEN_CALENDAR_POLICY,
            trading_calendar=tuple(bar.trade_date for bar in bars),
        )

        try:
            tensorized = tensorize_window_samples(samples)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        self.assertEqual(len(tensorized), 1)
        self.assertEqual(tensorized[0].target_stock_id, "000001.SZ")

    def test_v2_window_labels_tensorize_as_three_way_schema(self) -> None:
        as_of = date(2026, 1, 2)
        cutoff = datetime(2026, 1, 2, 18, 0)
        bars = tuple(
            _daily_bar("000001.SZ", as_of + timedelta(days=offset), 10 + offset)
            for offset in range(7)
        )
        v2_labels = tuple(
            _v2_label(as_of, cutoff, horizon, Decimal("0.01") * horizon)
            for horizon in (1, 3, 5)
        )
        samples = build_window_samples(
            records=(
                DragonTigerRecord(
                    trade_date=as_of,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    available_time=cutoff - timedelta(hours=1),
                ),
            ),
            bars=bars,
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=0.1,
            max_edge_age_days=20,
            v2_labels_by_key={(as_of, "000001.SZ"): v2_labels},
        )

        try:
            tensorized = tensorize_window_samples(samples)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        self.assertEqual(tensorized[0].label_schema_version, EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA)
        self.assertEqual(tensorized[0].labels, {1: 1, 3: 1, 5: 1})
        self.assertEqual(tensorized[0].returns, {1: 0.01, 3: 0.03, 5: 0.05})

    def test_bulk_window_tensorize_reuses_one_graph_for_same_as_of_date(self) -> None:
        as_of = date(2026, 1, 2)
        cutoff = datetime(2026, 1, 2, 18, 0)
        bars = tuple(
            bar
            for stock_id in ("000001.SZ", "000002.SZ")
            for bar in (
                _daily_bar(stock_id, as_of + timedelta(days=offset), 10 + offset)
                for offset in range(6)
            )
        )
        samples = build_window_samples(
            records=(
                DragonTigerRecord(
                    trade_date=as_of,
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    available_time=cutoff - timedelta(hours=1),
                ),
                DragonTigerRecord(
                    trade_date=as_of,
                    seat_id="seat-b",
                    stock_id="000002.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    available_time=cutoff - timedelta(hours=1),
                ),
            ),
            bars=bars,
            as_of_dates=(as_of,),
            target_stock_ids=("000001.SZ", "000002.SZ"),
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=0.1,
            max_edge_age_days=20,
            adjustment_policy=FROZEN_ADJUSTMENT_POLICY,
            calendar_policy=FROZEN_CALENDAR_POLICY,
            trading_calendar=tuple(as_of + timedelta(days=offset) for offset in range(6)),
        )

        try:
            tensorized = tensorize_window_samples(samples)
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        self.assertEqual(len(tensorized), 2)
        self.assertIs(tensorized[0].graph, tensorized[1].graph)


def _bars_for_correlation() -> tuple[DailyBar, ...]:
    closes = {
        "000001.SZ": (Decimal("10"), Decimal("11"), Decimal("13"), Decimal("16")),
        "000002.SZ": (Decimal("20"), Decimal("22"), Decimal("26"), Decimal("32")),
    }
    days = [date(2026, 1, day) for day in (10, 11, 12, 13)]
    bars: list[DailyBar] = []
    for stock_id, values in closes.items():
        for day, close in zip(days, values, strict=True):
            bars.append(_daily_bar(stock_id, day, int(close)))
    return tuple(bars)


def _daily_bar(stock_id: str, trade_date: date, close: int) -> DailyBar:
    value = Decimal(close)
    return DailyBar(
        trade_date=trade_date,
        stock_id=stock_id,
        open_price=value,
        high_price=value,
        low_price=value,
        close_price=value,
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="unit-test",
        available_time=datetime.combine(trade_date, datetime.min.time()),
        source_version="2026-01-fixture",
        volume_unit="share",
        amount_unit="cny",
    )


def _v2_label(
    as_of_date: date, cutoff_time: datetime, horizon: int, excess_return: Decimal
) -> StockLabelV2:
    start = as_of_date + timedelta(days=1)
    end = start + timedelta(days=horizon)
    return StockLabelV2(
        as_of_date=as_of_date,
        cutoff_time=cutoff_time,
        target_stock_id="000001.SZ",
        horizon=horizon,
        label_start_date=start,
        label_end_date=end,
        benchmark_id="000300.SH",
        price_policy=LABEL_V2_PRICE_POLICY,
        adjustment_policy=LABEL_V2_ADJUSTMENT_POLICY,
        calendar_policy=FROZEN_CALENDAR_POLICY,
        label_available_time=datetime.combine(end, datetime.max.time()),
        disposition="eligible",
        source_hash=f"{horizon}" * 64,
        stock_return=excess_return,
        benchmark_return=Decimal("0"),
        excess_return=excess_return,
        direction_label=1,
    )


if __name__ == "__main__":
    unittest.main()
