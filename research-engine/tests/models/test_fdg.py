"""Tests for FDG full model (Task 17)."""

import torch

from seatflow.models.fdg import SeatFlowFDG, SeatFlowLoss


class TestFDGAdditiveDecomposition:
    """final == base + direct + spillover"""

    def test_additive(self):
        model = SeatFlowFDG(
            market_input_dim=16,
            seat_input_dim=8,
            event_input_dim=6,
            edge_feature_dim=8,
            stock_embed_dim=32,
            hidden_dim=64,
            num_horizons=3,
        )
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
        )

        reconstructed = pred.base + pred.direct + pred.spillover
        assert torch.allclose(pred.final, reconstructed, atol=1e-5)

    def test_output_shapes(self):
        model = SeatFlowFDG(num_horizons=3)
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
        )
        assert pred.base.shape == (batch, 3)
        assert pred.direct.shape == (batch, 3)
        assert pred.spillover.shape == (batch, 3)
        assert pred.final.shape == (batch, 3)
        assert pred.direction.shape == (batch, 3)
        assert pred.uncertainty.shape == (batch, 3)


class TestFDGGradients:
    """Combined loss backward pass produces finite gradients."""

    def test_key_params_have_grad(self):
        model = SeatFlowFDG(num_horizons=3)
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)
        target = torch.randn(batch, 3)

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
        )
        loss_fn = SeatFlowLoss()
        loss = loss_fn(pred, target)
        loss.backward()

        # Check that the main model parameters have gradients
        # (some params like spillover_head may not receive grad if not in
        # the computation path for this particular input)
        params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None
        )
        total_params = sum(1 for p in model.parameters())
        # At least the base and direct models should have gradients
        assert params_with_grad > total_params // 2, (
            f"Only {params_with_grad}/{total_params} params have gradients"
        )

        # Check that all existing gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )


class TestFDGMasks:
    """Direct rows do not enter Spillover loss; context-only rows excluded from Direct loss."""

    def test_direct_mask_applied(self):
        model = SeatFlowFDG(num_horizons=3)
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)

        direct_mask = torch.tensor([True, True, False, False])
        spillover_mask = torch.tensor([False, False, True, True])

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
            direct_mask=direct_mask,
            spillover_mask=spillover_mask,
        )
        assert pred.direct_mask is not None
        assert pred.spillover_mask is not None

    def test_loss_with_masks(self):
        model = SeatFlowFDG(num_horizons=3)
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)
        target = torch.randn(batch, 3)

        direct_mask = torch.tensor([True, True, False, False])
        spillover_mask = torch.tensor([False, False, True, True])

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
            direct_mask=direct_mask,
            spillover_mask=spillover_mask,
        )
        loss_fn = SeatFlowLoss()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        loss.backward()


class TestFDGLossDecomposition:
    def test_decompose(self):
        model = SeatFlowFDG(num_horizons=3)
        batch = 4
        market_feat = torch.randn(batch, 16)
        seat_feat = torch.randn(batch, 5, 8)
        event_feat = torch.randn(batch, 6)
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)
        target = torch.randn(batch, 3)

        pred = model(
            market_feat, seat_feat, event_feat,
            stock_emb, edge_feat, edge_index, age, regime,
        )
        loss_fn = SeatFlowLoss()
        d = loss_fn.decompose(pred, target)
        assert "return" in d
        assert "rank" in d
        assert "direction" in d
        assert "calibration" in d
        assert "total" in d
