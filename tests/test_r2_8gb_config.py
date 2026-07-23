from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from seatflow.runtime.device import resolve_device


class R28GbConfigTest(unittest.TestCase):
    def test_memory_and_research_constraints_are_frozen(self) -> None:
        with Path("configs/experiments/r2_seatranklite_8gb.toml").open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["model"]["hidden_dim"], 64)
        self.assertEqual(config["model"]["num_layers"], 2)
        self.assertTrue(config["device"]["mixed_precision"])
        self.assertEqual(config["device"]["preferred_dtype"], "fp16")
        self.assertFalse(config["graph"]["enable_seat_cooccurrence"])
        self.assertFalse(config["graph"]["enable_concept_graph"])
        self.assertEqual(config["graph"]["hard_max_edges"], 100_000)
        self.assertEqual(config["training"]["seed_workers"], 1)
        self.assertEqual(config["training"]["seeds"], [2026, 2027, 2028, 2029, 2030])
        self.assertEqual(config["loss"]["edge_consistency_weight"], 0.0)
        self.assertAlmostEqual(
            sum(
                config["loss"][name]
                for name in (
                    "smooth_l1_weight",
                    "pairwise_rank_weight",
                    "pearson_weight",
                    "direction_weight",
                )
            ),
            1.0,
        )

    def test_auto_device_uses_cuda_on_this_gpu_host(self) -> None:
        import torch

        device = resolve_device("auto")
        self.assertEqual(device.type, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    unittest.main()
