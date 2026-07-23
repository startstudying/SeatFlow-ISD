from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder


class DuplicateStaticEdgesRemovedTest(unittest.TestCase):
    def test_r2_aggregates_trades_and_deduplicates_industry_relations(self) -> None:
        records = tuple(
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id="seat-a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=amount,
                industry="bank",
            )
            for amount in (100.0, 200.0)
        )
        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            enable_seat_cooccurrence=False,
            enable_concept_graph=False,
            add_reverse_relations=True,
            aggregate_duplicate_edges=True,
        ).build(records, date(2026, 1, 1))

        buys = [edge for edge in graph.edges if edge.relation == "buys"]
        belongs_to = [edge for edge in graph.edges if edge.relation == "belongs_to"]
        contains = [edge for edge in graph.edges if edge.relation == "contains"]

        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].attributes["amount"], 300.0)
        self.assertEqual(buys[0].attributes["trade_count"], 2.0)
        self.assertEqual(buys[0].attributes["amount_share_of_stock"], 1.0)
        self.assertEqual(buys[0].attributes["amount_share_of_seat"], 1.0)
        self.assertEqual(buys[0].attributes["net_buy_ratio"], 1.0)
        self.assertEqual(len(belongs_to), 1)
        self.assertEqual(len(contains), 1)


if __name__ == "__main__":
    unittest.main()
