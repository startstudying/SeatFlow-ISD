from __future__ import annotations

import unittest

from seatflow.training.seed_lifecycle import release_seed_resources


class SeedGpuMemoryReleasedTest(unittest.TestCase):
    def test_release_boundary_is_safe_on_cpu(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        report = release_seed_resources(
            model=model,
            optimizer=optimizer,
            graph=torch.ones(1),
            device="cpu",
        )

        self.assertFalse(report.available)
        self.assertEqual(report.allocated_bytes, 0)
        self.assertEqual(report.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
