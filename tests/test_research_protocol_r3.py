from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class ResearchProtocolR3Test(unittest.TestCase):
    def test_protocol_freezes_search_and_resource_limits(self) -> None:
        path = Path("configs/experiments/research_protocol_r3.toml")
        with path.open("rb") as stream:
            config = tomllib.load(stream)

        self.assertTrue(config["protocol"]["outer_r2_results_read_only"])
        self.assertEqual(config["protocol"]["maximum_new_configurations"], 5)
        self.assertEqual(config["inner_development"]["folds"], [0, 1])
        self.assertEqual(config["inner_development"]["seeds"], [2026, 2027, 2028])
        self.assertEqual(config["final_development"]["expected_runs"], 15)
        self.assertEqual(config["resource_limits"]["serial_seed_workers"], 1)
        self.assertLessEqual(config["resource_limits"]["maximum_peak_gpu_gib"], 7.2)
        self.assertLessEqual(config["resource_limits"]["maximum_parameters"], 5_000_000)
        self.assertFalse(config["task25"]["lambda_rank"]["allow_full_pair_matrix"])
        self.assertTrue(all(config["prohibited_first_round"].values()))


if __name__ == "__main__":
    unittest.main()
