from __future__ import annotations

import unittest
from datetime import date

from seatflow.evaluation.relation_mask import (
    mask_relations,
    mask_tensorized_relations,
    relation_types,
)
from seatflow.graph.tensorize import TensorizedGraph
from seatflow.graph.types import GraphEdge, GraphNode, HeteroGraphSnapshot, NodeType


class RelationMaskTest(unittest.TestCase):
    def test_masking_does_not_mutate_original_graph(self) -> None:
        edge = GraphEdge(
            source_type=NodeType.SEAT,
            source_id="seat-a",
            relation="buys",
            target_type=NodeType.STOCK,
            target_id="000001.SZ",
            weight=1.0,
            event_date=date(2026, 1, 2),
            attributes={"amount": 1.0},
        )
        graph = HeteroGraphSnapshot(
            as_of_date=date(2026, 1, 2),
            nodes=(
                GraphNode(NodeType.SEAT, "seat-a"),
                GraphNode(NodeType.STOCK, "000001.SZ"),
            ),
            edges=(edge,),
        )
        masked = mask_relations(graph, exclude=("buys",))

        self.assertNotEqual(masked, graph)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(masked.edges), 0)
        self.assertEqual(graph.edges[0].relation, "buys")
        self.assertEqual(graph.edges[0].event_date, date(2026, 1, 2))

    def test_mask_uses_inclusive_relation_filter(self) -> None:
        graph = HeteroGraphSnapshot(
            as_of_date=date(2026, 1, 2),
            nodes=(GraphNode(NodeType.SEAT, "seat-a"), GraphNode(NodeType.STOCK, "000001.SZ")),
            edges=(
                GraphEdge(
                    source_type=NodeType.SEAT,
                    source_id="seat-a",
                    relation="buys",
                    target_type=NodeType.STOCK,
                    target_id="000001.SZ",
                    weight=1.0,
                    event_date=date(2026, 1, 2),
                    attributes={},
                ),
                GraphEdge(
                    source_type=NodeType.SEAT,
                    source_id="seat-b",
                    relation="sells",
                    target_type=NodeType.STOCK,
                    target_id="000001.SZ",
                    weight=1.0,
                    event_date=date(2026, 1, 2),
                    attributes={},
                ),
            ),
        )
        kept = mask_relations(graph, include=("sells",))
        self.assertEqual(tuple(edge.relation for edge in kept.edges), ("sells",))
        self.assertEqual(relation_types(graph), ("buys", "sells"))

    def test_tensorized_mask_clones_before_emptying_edges(self) -> None:
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["seat"].x = torch.ones((1, 1))
        data["stock"].x = torch.ones((1, 1))
        edge_type = ("seat", "buys", "stock")
        data[edge_type].edge_index = torch.tensor([[0], [0]])
        data[edge_type].edge_attr = torch.ones((1, 2))
        data[edge_type].edge_weight = torch.ones(1)
        graph = TensorizedGraph(data=data, metadata={"edge_shapes": {}})

        masked = mask_tensorized_relations(graph, "without_seat_stock")

        self.assertEqual(graph.data[edge_type].edge_index.shape[1], 1)
        self.assertEqual(masked.data[edge_type].edge_index.shape[1], 0)
        self.assertIsNot(masked.data, graph.data)


if __name__ == "__main__":
    unittest.main()
