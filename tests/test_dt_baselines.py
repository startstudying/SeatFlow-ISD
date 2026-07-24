from __future__ import annotations

import unittest
from datetime import date, timedelta
from math import log1p

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.evaluation.dt_baselines import (
    DT_BASELINE_FEATURE_NAMES,
    DT_BASELINE_NAME,
    _feature_vector,
    run_dt_graph_baseline,
)
from seatflow.graph.tensorize import TensorizedGraph


class DTBaselineTest(unittest.TestCase):
    def test_baseline_uses_dt_labels_and_graph_contract(self) -> None:
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ModuleNotFoundError as error:
            self.skipTest(str(error))

        data = HeteroData()
        data["seat"].x = torch.tensor(
            [[1.0, 0.5, 0.1, 0.9, 0.6, 0.02, 0.03, 2.0]], dtype=torch.float32
        )
        data["stock"].x = torch.ones((1, 1), dtype=torch.float32)
        edge_type = ("seat", "buys", "stock")
        data[edge_type].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data[edge_type].edge_attr = torch.tensor(
            [[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32
        )
        data[edge_type].edge_weight = torch.tensor([1.0], dtype=torch.float32)
        metadata = {
            "edge_attr_schema": {
                "seat::buys::stock": [
                    "amount",
                    "direction",
                    "edge_gate_input",
                    "graph_edge_weight",
                    "seat_profile_available",
                    "time_decay",
                ]
            },
            "feature_schema": {
                "seat": list(DT_BASELINE_FEATURE_NAMES[4:]),
            },
        }
        graph = TensorizedGraph(data=data, metadata=metadata)

        def sample(index: int, label: int) -> TensorizedWindowSample:
            as_of = date(2024, 1, 2) + timedelta(days=index)
            return TensorizedWindowSample(
                sample_id=f"{as_of.isoformat()}::000001.SZ",
                source_hash="a" * 64,
                target_stock_id="000001.SZ",
                target_node_index=0,
                graph=graph,
                labels={1: label, 3: label, 5: label},
                returns={1: 0.02 if label else -0.01, 3: 0.02, 5: 0.03},
                as_of_date=as_of,
                label_end_dates={1: as_of + timedelta(days=1)},
            )

        evaluation = run_dt_graph_baseline(
            (sample(0, 0), sample(1, 1)),
            (sample(2, 1),),
            seed=2026,
            fold_id="fold_0",
            epochs=2,
        )

        self.assertEqual(evaluation.baseline_name, DT_BASELINE_NAME)
        self.assertEqual(evaluation.fold_id, "fold_0")
        self.assertEqual(evaluation.metrics.sample_count, 1)
        self.assertEqual(len(evaluation.scores), 1)
        self.assertTrue(0.0 <= evaluation.scores[0].score <= 1.0)

    def test_feature_vector_aggregates_raw_amounts_not_logged_edges(self) -> None:
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ModuleNotFoundError as error:
            self.skipTest(str(error))

        data = HeteroData()
        data["seat"].x = torch.zeros((2, 8), dtype=torch.float32)
        data["stock"].x = torch.ones((1, 1), dtype=torch.float32)
        edge_type = ("seat", "buys", "stock")
        data[edge_type].edge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
        data[edge_type].edge_attr = torch.tensor(
            [
                [log1p(100.0), 1.0, 0.0, 0.0],
                [log1p(300.0), 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        graph = TensorizedGraph(
            data=data,
            metadata={
                "edge_attr_schema": {
                    "seat::buys::stock": [
                        "amount",
                        "direction",
                        "seat_profile_available",
                        "graph_edge_weight",
                    ]
                },
                "feature_schema": {"seat": list(DT_BASELINE_FEATURE_NAMES[4:])},
            },
        )
        sample = TensorizedWindowSample(
            sample_id="2026-01-01::000001.SZ",
            source_hash="a" * 64,
            target_stock_id="000001.SZ",
            target_node_index=0,
            graph=graph,
            labels={1: 1},
            returns={1: 0.02},
            as_of_date=date(2026, 1, 1),
        )

        features = _feature_vector(sample)

        self.assertAlmostEqual(features[0], 400.0, places=3)
        self.assertAlmostEqual(features[2], 400.0, places=3)
        self.assertEqual(features[3], 2.0)


if __name__ == "__main__":
    unittest.main()
