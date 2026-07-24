from __future__ import annotations

import unittest

from seatflow.training.ranking_losses import SampledPairwiseRankLoss


class PairwiseLossSameDateOnlyTest(unittest.TestCase):
    def test_single_stock_dates_do_not_form_cross_date_pairs(self) -> None:
        import torch

        predictions = torch.tensor([1.0, -1.0], requires_grad=True)
        targets = torch.tensor([-1.0, 1.0])
        loss_function = SampledPairwiseRankLoss()
        loss = loss_function(predictions, targets, ("2026-01-01", "2026-01-02"))
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(loss_function.last_pair_count_by_date["2026-01-01"], 0)
        self.assertEqual(loss_function.last_pair_count_by_date["2026-01-02"], 0)


if __name__ == "__main__":
    unittest.main()
