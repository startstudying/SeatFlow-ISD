from __future__ import annotations

import unittest

from seatflow.training.ranking_losses import LambdaRankAtKLoss, compute_ranking_loss


class LambdaRankAt5Test(unittest.TestCase):
    def test_pair_count_is_sparse_bounded_and_gradient_is_finite(self) -> None:
        import torch

        size = 200
        predictions = torch.linspace(-1, 1, size, requires_grad=True)
        targets = torch.linspace(0, 1, size)
        loss_function = LambdaRankAtKLoss(k=5, max_pairs_per_date=1024, seed=7)
        loss = loss_function(predictions, targets, ("2026-01-01",) * size)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertLessEqual(loss_function.last_pair_count_by_date["2026-01-01"], 1024)
        self.assertGreater(loss_function.last_pair_count_by_date["2026-01-01"], 0)
        self.assertTrue(bool(torch.isfinite(predictions.grad).all()))

    def test_l2_composite_uses_separate_regression_and_rank_targets(self) -> None:
        import torch

        outputs = {
            "residual_return": torch.tensor([0.2, -0.1, 0.3], requires_grad=True),
            "rank_score": torch.tensor([0.1, 0.3, -0.2], requires_grad=True),
            "direction_logit": torch.tensor([0.0, 0.0, 0.0], requires_grad=True),
        }
        loss, components = compute_ranking_loss(
            outputs,
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            ("2026-01-01",) * 3,
            objective="L2",
            ranking_targets=torch.tensor([0.0, 0.5, 1.0]),
        )
        loss.backward()
        self.assertIn("pairwise_rank", components)
        self.assertTrue(bool(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()
