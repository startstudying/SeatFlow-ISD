from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from seatflow.training.r3_task25 import (
    build_inner_folds,
    create_immutable_run_manifest,
    legalize_r3_graph,
)


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

    def test_two_fold_manifest_is_exact_and_refuses_overwrite(self) -> None:
        samples = tuple(
            SimpleNamespace(
                as_of_date=date(2020, 1, 1) + timedelta(days=index),
                label_end_dates={1: date(2020, 1, 2) + timedelta(days=index)},
                sample_id=f"sample-{index}",
            )
            for index in range(324)
        )
        folds = build_inner_folds(
            samples,
            minimum_train_dates=320,
            purge_days=1,
            embargo_days=1,
        )
        self.assertEqual(len(folds), 2)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "config.toml"
            config.write_text("[freeze]\nimmutable=true\n", encoding="utf-8")
            path = create_immutable_run_manifest(
                samples,
                config_path=config,
                output_root=root / "artifacts",
            )
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["expected_run_count"], 18)
            self.assertEqual(len({row["run_id"] for row in payload["run_order"]}), 18)
            with self.assertRaises(FileExistsError):
                create_immutable_run_manifest(
                    samples,
                    config_path=config,
                    output_root=root / "artifacts",
                )


if __name__ == "__main__":
    unittest.main()
