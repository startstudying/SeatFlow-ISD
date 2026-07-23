from __future__ import annotations

import unittest

from seatflow.training.r3_task25 import legalize_r3_graph


class R3Task25PreflightTest(unittest.TestCase):
    def test_legacy_cooccurrence_is_removed_and_reverse_edges_are_added(self) -> None:
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            self.skipTest("PyTorch Geometric is required")
        from seatflow.graph.tensorize import TensorizedGraph

        data = HeteroData()
        data["seat"].x = torch.ones((2, 1))
        data["stock"].x = torch.ones((2, 1))
        data["seat", "buys", "stock"].edge_index = torch.tensor([[0, 1], [0, 1]])
        data["seat", "buys", "stock"].edge_attr = torch.ones((2, 3))
        data["seat", "cooccurs_with", "seat"].edge_index = torch.tensor([[0], [1]])
        data["seat", "cooccurs_with", "seat"].edge_attr = torch.ones((1, 1))

        legalized = legalize_r3_graph(TensorizedGraph(data=data, metadata={}))

        self.assertNotIn(("seat", "cooccurs_with", "seat"), legalized.data.edge_types)
        self.assertIn(("stock", "rev_buys", "seat"), legalized.data.edge_types)
        self.assertTrue(
            torch.equal(
                legalized.data["stock", "rev_buys", "seat"].edge_index,
                torch.tensor([[0, 1], [0, 1]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
