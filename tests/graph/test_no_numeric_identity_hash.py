from __future__ import annotations

import unittest
from datetime import date

from seatflow.graph.tensorize import derive_frozen_graph_schema
from seatflow.graph.types import GraphEdge, GraphNode, HeteroGraphSnapshot, NodeType


class NoNumericIdentityHashTest(unittest.TestCase):
    def test_every_prohibited_hash_is_rejected_from_edge_mlp_schema(self) -> None:
        prohibited = (
            "stock_id_hash",
            "seat_id_hash",
            "side_hash",
            "window_start_hash",
            "window_end_hash",
            "stock_code_hash",
            "seat_code_hash",
        )
        for name in prohibited:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "entity identifiers"
            ):
                derive_frozen_graph_schema((_snapshot_with_attribute(name),))


def _snapshot_with_attribute(name: str) -> HeteroGraphSnapshot:
    return HeteroGraphSnapshot(
        as_of_date=date(2026, 1, 1),
        nodes=(
            GraphNode(NodeType.SEAT, "seat-a"),
            GraphNode(NodeType.STOCK, "000001.SZ"),
        ),
        edges=(
            GraphEdge(
                NodeType.SEAT,
                "seat-a",
                "buys",
                NodeType.STOCK,
                "000001.SZ",
                1.0,
                date(2026, 1, 1),
                {name: 1.0},
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
