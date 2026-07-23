from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder


class SeatCooccurrenceDisabledTest(unittest.TestCase):
    def test_disabled_graph_has_no_seat_to_seat_edges(self) -> None:
        records = tuple(
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id=f"seat-{index}",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=100.0 + index,
            )
            for index in range(10)
        )
        graph = DynamicGraphBuilder(
            decay_lambda=0.1,
            max_edge_age_days=20,
            enable_seat_cooccurrence=False,
        ).build(records, date(2026, 1, 1))

        self.assertFalse(any(edge.relation == "cooccurs_with" for edge in graph.edges))
        self.assertEqual(len(graph.edges), len(records))


if __name__ == "__main__":
    unittest.main()
