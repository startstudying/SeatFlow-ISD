from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import derive_frozen_graph_schema
from seatflow.graph.types import GraphEdge, GraphNode, HeteroGraphSnapshot, NodeType


class NoEntityIdInContinuousFeaturesTest(unittest.TestCase):
    def test_builder_does_not_emit_hash_features(self) -> None:
        graph = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
            (
                DragonTigerRecord(
                    trade_date=date(2026, 1, 1), seat_id="seat-a", stock_id="000001.SZ",
                    side=TradeSide.BUY, amount=100.0,
                ),
            ),
            date(2026, 1, 1),
        )
        attribute_keys = {key for edge in graph.edges for key in edge.attributes}
        self.assertFalse(any(key.endswith("_hash") for key in attribute_keys))

    def test_entity_hash_in_edge_attributes_fails_closed(self) -> None:
        graph = HeteroGraphSnapshot(
            as_of_date=date(2026, 1, 1),
            nodes=(GraphNode(NodeType.SEAT, "seat-a"), GraphNode(NodeType.SEAT, "seat-b")),
            edges=(
                GraphEdge(
                    NodeType.SEAT, "seat-a", "cooccurs_with", NodeType.SEAT, "seat-b", 1.0,
                    date(2026, 1, 1), {"stock_id_hash": 123.0},
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "entity identifiers"):
            derive_frozen_graph_schema((graph,))


if __name__ == "__main__":
    unittest.main()
