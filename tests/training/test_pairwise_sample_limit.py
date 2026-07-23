from __future__ import annotations

import unittest

from seatflow.training.ranking_losses import SampledPairwiseRankLoss


class PairwiseSampleLimitTest(unittest.TestCase):
    def test_large_date_never_exceeds_pair_limit(self) -> None:
        import torch

        size = 200
        predictions = torch.linspace(-1.0, 1.0, size, requires_grad=True)
        targets = torch.arange(size, dtype=torch.float32)
        loss_function = SampledPairwiseRankLoss(max_pairs_per_date=1024, seed=7)
        loss = loss_function(predictions, targets, ("2026-01-01",) * size)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(loss_function.last_pair_count_by_date["2026-01-01"], 1024)


if __name__ == "__main__":
    unittest.main()
