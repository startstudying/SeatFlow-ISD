from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder


class ReverseRelationsExistTest(unittest.TestCase):
    def test_r2_bidirectional_trade_and_industry_relations_exist(self) -> None:
        records = (
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id="seat-buy",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=100.0,
                industry="bank",
                concepts=("disabled-concept",),
            ),
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id="seat-sell",
                stock_id="000002.SZ",
                side=TradeSide.SELL,
                amount=50.0,
                industry="bank",
            ),
        )
        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            enable_seat_cooccurrence=False,
            enable_concept_graph=False,
            add_reverse_relations=True,
        ).build(records, date(2026, 1, 1))
        relations = {
            (edge.source_type.value, edge.relation, edge.target_type.value)
            for edge in graph.edges
        }

        self.assertIn(("seat", "buys", "stock"), relations)
        self.assertIn(("stock", "bought_by", "seat"), relations)
        self.assertIn(("seat", "sells", "stock"), relations)
        self.assertIn(("stock", "sold_by", "seat"), relations)
        self.assertIn(("stock", "belongs_to", "industry"), relations)
        self.assertIn(("industry", "contains", "stock"), relations)
        self.assertFalse(any("concept" in relation for relation in relations))


if __name__ == "__main__":
    unittest.main()
