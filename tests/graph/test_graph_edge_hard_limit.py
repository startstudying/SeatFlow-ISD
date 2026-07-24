from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.stock_correlation import build_stock_correlation_edges


class GraphEdgeHardLimitTest(unittest.TestCase):
    def test_hard_limit_fails_with_relation_statistics(self) -> None:
        records = tuple(
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id=f"seat-{index}",
                stock_id=f"{index:06d}.SZ",
                side=TradeSide.BUY,
                amount=100.0,
            )
            for index in range(3)
        )
        builder = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            enable_seat_cooccurrence=False,
            add_reverse_relations=True,
            soft_max_edges=4,
            hard_max_edges=5,
        )

        with self.assertRaisesRegex(ValueError, "relations="):
            builder.build(records, date(2026, 1, 1))

        self.assertIsNotNone(builder.last_build_stats)
        self.assertEqual(builder.last_build_stats["total_edge_count"], 6)

    def test_edge_statistics_are_emitted_without_truncation(self) -> None:
        builder = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            enable_seat_cooccurrence=False,
            soft_max_edges=10,
            hard_max_edges=20,
        )
        graph = builder.build(
            (
                DragonTigerRecord(
                    trade_date=date(2026, 1, 1),
                    seat_id="seat-a",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                ),
            ),
            date(2026, 1, 1),
        )

        self.assertEqual(builder.last_build_stats["total_edge_count"], len(graph.edges))
        self.assertFalse(builder.last_build_stats["soft_limit_exceeded"])

    def test_stock_correlation_is_topk_sparse_per_source(self) -> None:
        bars = tuple(
            DailyBar(
                stock_id=stock_id,
                trade_date=date(2026, 1, day),
                open_price=Decimal(str(day * multiplier)),
                high_price=Decimal(str(day * multiplier + 1)),
                low_price=Decimal(str(day * multiplier - 1)),
                close_price=Decimal(str(day * multiplier)),
                volume=Decimal("100"),
                amount=Decimal("1000"),
                source_name="fixture",
            )
            for stock_id, multiplier in (
                ("000001.SZ", 1),
                ("000002.SZ", 2),
                ("000003.SZ", 3),
            )
            for day in range(1, 6)
        )
        edges = build_stock_correlation_edges(
            bars,
            as_of_date=date(2026, 1, 5),
            min_overlap=3,
            min_abs_correlation=0.0,
            positive_topk=1,
            negative_topk=0,
        )
        counts: dict[str, int] = {}
        for edge in edges:
            counts[edge.source_id] = counts.get(edge.source_id, 0) + 1

        self.assertTrue(counts)
        self.assertTrue(all(count <= 1 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
