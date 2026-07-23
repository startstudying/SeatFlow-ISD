from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from seatflow.training.seed_lifecycle import save_seed_artifacts


class SeedStateSavedOnCpuTest(unittest.TestCase):
    def test_checkpoint_tensors_are_all_cpu_resident(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        model = torch.nn.Linear(2, 1)
        with TemporaryDirectory() as temporary_directory:
            path = save_seed_artifacts(
                temporary_directory,
                fold_id=0,
                seed=2026,
                model=model,
                predictions={"000001.SZ": 0.5},
            )
            state = torch.load(path, weights_only=True)

        self.assertTrue(state)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in state.values()))

    def test_existing_seed_artifacts_are_not_overwritten(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        model = torch.nn.Linear(2, 1)
        with TemporaryDirectory() as temporary_directory:
            save_seed_artifacts(
                temporary_directory,
                fold_id=0,
                seed=2026,
                model=model,
                predictions={},
            )
            with self.assertRaises(FileExistsError):
                save_seed_artifacts(
                    temporary_directory,
                    fold_id=0,
                    seed=2026,
                    model=model,
                    predictions={},
                )


if __name__ == "__main__":
    unittest.main()
