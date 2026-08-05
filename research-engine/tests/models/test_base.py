"""Tests for Base model (Task 15)."""

import torch
import pytest

from seatflow.models.base import BaseMarketModel, BaseLoss, BasePrediction


class TestBaseModelShape:
    def test_forward_shape(self):
        model = BaseMarketModel(input_dim=16, hidden_dim=64, num_horizons=3)
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 3)

    def test_output_dim(self):
        model = BaseMarketModel(input_dim=16, num_horizons=3)
        assert model.output_dim == 3

    def test_base_model_no_seat_input(self):
        """Base model must NOT accept seat or event tensors."""
        model = BaseMarketModel(input_dim=16)
        # Only market features should be accepted
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape[0] == 4

    def test_single_sample(self):
        model = BaseMarketModel(input_dim=16, num_horizons=3)
        x = torch.randn(1, 16)
        out = model(x)
        assert out.shape == (1, 3)


class TestBaseLoss:
    def test_huber_loss(self):
        loss_fn = BaseLoss(huber_delta=1.0)
        pred = torch.tensor([[0.1, -0.05, 0.2]])
        target = torch.tensor([[0.12, -0.03, 0.18]])
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_decompose(self):
        loss_fn = BaseLoss()
        pred = torch.tensor([[0.1, -0.05, 0.2], [0.05, 0.1, -0.1]])
        target = torch.tensor([[0.12, -0.03, 0.18], [0.06, 0.08, -0.12]])
        d = loss_fn.decompose(pred, target)
        assert "huber" in d
        assert "rank" in d
        assert "total" in d

    def test_zero_prediction(self):
        loss_fn = BaseLoss()
        pred = torch.zeros(4, 3)
        target = torch.zeros(4, 3)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestBasePrediction:
    def test_prediction_fields(self):
        p = BasePrediction(
            stock_id="stock-1",
            prediction_date="2025-01-01",
            horizon=1,
            raw_excess_return=0.05,
            confidence=0.9,
            path="Base",
        )
        assert p.stock_id == "stock-1"
        assert p.raw_excess_return == 0.05
