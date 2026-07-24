import unittest
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.temporal import time_decay
from seatflow.graph.types import NodeType
from seatflow.labels import StockLabel


class TemporalWeightTest(unittest.TestCase):
    def test_weight_decreases_with_age(self) -> None:
        recent = time_decay(100.0, delta_days=1, decay_lambda=0.1)
        old = time_decay(100.0, delta_days=5, decay_lambda=0.1)

        self.assertGreater(recent, old)


class DynamicGraphBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.as_of_date = date(2026, 1, 15)
        self.builder = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20)

    def test_builds_five_node_types(self) -> None:
        record = DragonTigerRecord(
            trade_date=self.as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            industry="banking",
            concepts=("fintech",),
            event_reason="daily_deviation",
        )

        graph = self.builder.build([record], self.as_of_date)
        node_types = {node.node_type for node in graph.nodes}

        self.assertEqual(node_types, set(NodeType))
        self.assertEqual(len(graph.edges), 4)

    def test_rejects_future_records(self) -> None:
        future_record = DragonTigerRecord(
            trade_date=self.as_of_date + timedelta(days=1),
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
        )

        with self.assertRaisesRegex(ValueError, "leak future information"):
            self.builder.build([future_record], self.as_of_date)

    def test_ignores_expired_edges(self) -> None:
        expired_record = DragonTigerRecord(
            trade_date=self.as_of_date - timedelta(days=21),
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.SELL,
            amount=100.0,
        )

        graph = self.builder.build([expired_record], self.as_of_date)

        self.assertEqual(graph.nodes, ())
        self.assertEqual(graph.edges, ())

    def test_rejects_negative_decay(self) -> None:
        with self.assertRaisesRegex(ValueError, "decay_lambda"):
            DynamicGraphBuilder(decay_lambda=-0.1, max_edge_age_days=20)

    def test_excludes_records_not_available_at_cutoff(self) -> None:
        record = DragonTigerRecord(
            trade_date=self.as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            available_time=datetime.combine(self.as_of_date, time(18, 0)),
        )

        graph = self.builder.build(
            [record], self.as_of_date, cutoff_time=datetime.combine(self.as_of_date, time(17, 0))
        )

        self.assertEqual(graph.nodes, ())
        self.assertEqual(graph.edges, ())

    def test_rejects_mixed_timezone_awareness(self) -> None:
        record = DragonTigerRecord(
            trade_date=self.as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            available_time=datetime.combine(self.as_of_date, time(18, 0), UTC),
        )

        with self.assertRaisesRegex(ValueError, "timezone awareness"):
            self.builder.build(
                [record],
                self.as_of_date,
                cutoff_time=datetime.combine(self.as_of_date, time(17, 0)),
            )

    def test_adds_only_matured_seat_profile_features(self) -> None:
        cutoff = datetime.combine(self.as_of_date, time(18, 0))
        record = DragonTigerRecord(
            trade_date=date(2026, 1, 10),
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            available_time=datetime(2026, 1, 10, 18, 0),
        )
        matured = _label(
            as_of_date=date(2026, 1, 10),
            stock_id="000001.SZ",
            available_time=datetime(2026, 1, 13, 18, 0),
            future_return=Decimal("0.05"),
        )
        immature = _label(
            as_of_date=date(2026, 1, 10),
            stock_id="000001.SZ",
            available_time=datetime(2026, 1, 16, 18, 0),
            future_return=Decimal("-0.10"),
        )

        graph = self.builder.build(
            [record],
            self.as_of_date,
            cutoff_time=cutoff,
            matured_labels=(matured, immature),
        )

        seat = next(node for node in graph.nodes if node.node_type == NodeType.SEAT)
        self.assertEqual(seat.features["seat_profile_available"], 1.0)
        self.assertEqual(seat.features["seat_profile_event_count"], 1.0)
        self.assertAlmostEqual(seat.features["seat_profile_avg_signed_return"], 0.05)

    def test_skips_seat_profile_when_no_matured_label_exists(self) -> None:
        cutoff = datetime.combine(self.as_of_date, time(18, 0))
        record = DragonTigerRecord(
            trade_date=date(2026, 1, 10),
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            available_time=datetime(2026, 1, 10, 18, 0),
        )
        immature = _label(
            as_of_date=date(2026, 1, 10),
            stock_id="000001.SZ",
            available_time=datetime(2026, 1, 16, 18, 0),
            future_return=Decimal("0.05"),
        )

        graph = self.builder.build(
            [record],
            self.as_of_date,
            cutoff_time=cutoff,
            matured_labels=(immature,),
        )

        seat = next(node for node in graph.nodes if node.node_type == NodeType.SEAT)
        self.assertNotIn("seat_profile_available", seat.features)
        trade_edge = next(edge for edge in graph.edges if edge.relation == "buys")
        self.assertEqual(trade_edge.attributes["seat_profile_available"], 0.0)

    def test_adds_seat_cooccurrence_edges_with_auditable_attrs(self) -> None:
        records = (
            DragonTigerRecord(
                trade_date=self.as_of_date,
                seat_id="seat-a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=100.0,
            ),
            DragonTigerRecord(
                trade_date=self.as_of_date,
                seat_id="seat-b",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=50.0,
            ),
        )

        graph = self.builder.build(records, self.as_of_date)
        cooccur = [edge for edge in graph.edges if edge.relation == "cooccurs_with"]

        self.assertEqual(len(cooccur), 2)
        self.assertEqual(cooccur[0].attributes["direction"], -1.0)
        self.assertEqual(cooccur[0].attributes["amount"], 50.0)
        self.assertIn("time_decay", cooccur[0].attributes)

    def test_cooccurrence_cap_keeps_largest_records_deterministically(self) -> None:
        records = tuple(
            DragonTigerRecord(
                trade_date=self.as_of_date,
                seat_id=f"seat-{index}",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=float(index),
            )
            for index in range(1, 5)
        )
        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            cooccurrence_max_records_per_stock=2,
        ).build(records, self.as_of_date)

        cooccur = [edge for edge in graph.edges if edge.relation == "cooccurs_with"]
        self.assertEqual(len(cooccur), 2)
        self.assertEqual({edge.source_id for edge in cooccur}, {"seat-3", "seat-4"})

    def test_adds_positive_and_negative_stock_correlation_edges(self) -> None:
        bars = _bars_for_correlation()

        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            correlation_min_abs=0.9,
        ).build([], self.as_of_date, bars=bars)
        relations = {edge.relation for edge in graph.edges}

        self.assertIn("positively_correlated_with", relations)
        self.assertIn("negatively_correlated_with", relations)
        for edge in graph.edges:
            self.assertIn("correlation", edge.attributes)
            self.assertIn("edge_gate_input", edge.attributes)

    def test_stock_features_use_as_of_bars_without_adding_unrelated_bar_nodes(self) -> None:
        record = DragonTigerRecord(
            trade_date=self.as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
        )
        graph = self.builder.build(
            [record],
            self.as_of_date,
            bars=(
                _daily_bar(self.as_of_date - timedelta(days=1), "000001.SZ", 10),
                _daily_bar(self.as_of_date, "000001.SZ", 11),
                _daily_bar(self.as_of_date, "000002.SZ", 99),
            ),
        )

        stock_nodes = [node for node in graph.nodes if node.node_type == NodeType.STOCK]
        self.assertEqual([node.node_id for node in stock_nodes], ["000001.SZ"])
        self.assertEqual(stock_nodes[0].features["stock_bar_count"], 2.0)
        self.assertAlmostEqual(stock_nodes[0].features["stock_return_1d"], 0.1)

    def test_stock_features_reject_future_and_honor_cutoff(self) -> None:
        record = DragonTigerRecord(
            trade_date=self.as_of_date,
            seat_id="seat-a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=100.0,
            available_time=datetime.combine(self.as_of_date, time(16, 0)),
        )
        cutoff = datetime.combine(self.as_of_date, time(17, 0))
        graph = self.builder.build(
            [record],
            self.as_of_date,
            cutoff_time=cutoff,
            bars=(
                _daily_bar(
                    self.as_of_date,
                    "000001.SZ",
                    10,
                    available_time=datetime.combine(self.as_of_date, time(16, 0)),
                ),
                _daily_bar(
                    self.as_of_date,
                    "000001.SZ",
                    20,
                    available_time=datetime.combine(self.as_of_date, time(18, 0)),
                ),
            ),
        )
        stock = next(node for node in graph.nodes if node.node_type == NodeType.STOCK)
        self.assertEqual(stock.features["stock_bar_count"], 1.0)
        self.assertEqual(stock.features["stock_close_to_open"], 0.0)

        with self.assertRaisesRegex(ValueError, "leak future information"):
            self.builder.build(
                [record],
                self.as_of_date,
                bars=(_daily_bar(self.as_of_date + timedelta(days=1), "000001.SZ", 10),),
            )

    def test_bar_generator_feeds_features_and_correlation_once_materialized(self) -> None:
        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            correlation_min_abs=0.9,
        ).build([], self.as_of_date, bars=(bar for bar in _bars_for_correlation()))

        self.assertIn("positively_correlated_with", {edge.relation for edge in graph.edges})
        stock = next(node for node in graph.nodes if node.node_id == "000001.SZ")
        self.assertIn("stock_bar_count", stock.features)

def _label(
    *,
    as_of_date: date,
    stock_id: str,
    available_time: datetime,
    future_return: Decimal,
) -> StockLabel:
    return StockLabel(
        as_of_date=as_of_date,
        cutoff_time=datetime.combine(as_of_date, time(18, 0)),
        target_stock_id=stock_id,
        horizon=1,
        label_start_date=as_of_date + timedelta(days=1),
        label_end_date=as_of_date + timedelta(days=1),
        base_close=Decimal("10"),
        future_close=Decimal("10") * (Decimal("1") + future_return),
        future_return=future_return,
        direction_label=1 if future_return > 0 else 0,
        label_available_time=available_time,
        adjustment_policy="raw_close_fixture_only",
        calendar_policy="observed_trading_days_fixture_only",
        base_source_name="fixture",
        future_source_name="fixture",
    )


def _bars_for_correlation() -> tuple[DailyBar, ...]:
    closes = {
        "000001.SZ": (Decimal("10"), Decimal("11"), Decimal("13"), Decimal("16")),
        "000002.SZ": (Decimal("20"), Decimal("22"), Decimal("26"), Decimal("32")),
        "000003.SZ": (Decimal("30"), Decimal("27"), Decimal("23"), Decimal("18")),
    }
    days = [date(2026, 1, day) for day in (10, 11, 12, 13)]
    bars: list[DailyBar] = []
    for stock_id, stock_closes in closes.items():
        for day, close in zip(days, stock_closes, strict=True):
            bars.append(
                DailyBar(
                    trade_date=day,
                    stock_id=stock_id,
                    open_price=close,
                    high_price=close,
                    low_price=close,
                    close_price=close,
                    volume=Decimal("1"),
                    amount=Decimal("1"),
                    source_name="fixture",
                )
            )
    return tuple(bars)


def _daily_bar(
    trade_date: date,
    stock_id: str,
    close: int,
    *,
    available_time: datetime | None = None,
) -> DailyBar:
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
        source_name="fixture",
        available_time=available_time,
    )


if __name__ == "__main__":
    unittest.main()
