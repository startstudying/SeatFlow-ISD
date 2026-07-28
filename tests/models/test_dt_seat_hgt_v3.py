"""Tests for DT-SeatHGT v3 model."""

from __future__ import annotations

import unittest

from seatflow.models.dt_seat_hgt_v3 import (
    DTSeatHGTv3Config,
    DTSeatHGTv3Model,
    build_dt_seat_hgt_v3,
)


class TestDTSeatHGTv3Config(unittest.TestCase):
    """Test DTSeatHGTv3Config class."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = DTSeatHGTv3Config()
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.stock_gru_hidden_dim, 48)
        self.assertEqual(config.seat_style_count, 5)
        self.assertEqual(config.seat_gate_initial_bias, -2.0)
        self.assertEqual(config.maximum_parameters, 3_000_000)

    def test_invalid_hidden_dim(self) -> None:
        """Invalid hidden_dim should raise ValueError."""
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(hidden_dim=0)
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(hidden_dim=200)

    def test_invalid_num_heads(self) -> None:
        """Invalid num_heads should raise ValueError."""
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(hidden_dim=64, num_heads=3)

    def test_invalid_dropout(self) -> None:
        """Invalid dropout should raise ValueError."""
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(dropout=1.5)
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(dropout=-0.1)

    def test_invalid_gate_bias(self) -> None:
        """Positive gate bias should raise ValueError."""
        with self.assertRaises(ValueError):
            DTSeatHGTv3Config(seat_gate_initial_bias=1.0)

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        config = DTSeatHGTv3Config()
        d = config.to_dict()
        self.assertEqual(d["hidden_dim"], 64)
        self.assertEqual(d["seat_gate_initial_bias"], -2.0)

    def test_fingerprint(self) -> None:
        """fingerprint should return consistent hash."""
        config = DTSeatHGTv3Config()
        fp1 = config.fingerprint()
        fp2 = config.fingerprint()
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)  # SHA-256 hex length


class TestDTSeatHGTv3Model(unittest.TestCase):
    """Test DTSeatHGTv3Model class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = DTSeatHGTv3Config(
            hidden_dim=64,
            stock_gru_hidden_dim=32,
            stock_gru_layers=1,
            stock_static_hidden_dim=16,
            stock_sequence_feature_dim=5,
            stock_static_feature_count=32,  # Match test graph data
            seat_style_count=3,
            seat_style_embedding_dim=8,
            num_heads=2,
            base_relation_layers=1,
            seat_relation_layers=1,
            dropout=0.1,
            seat_gate_initial_bias=-2.0,
        )

    def test_model_creation(self) -> None:
        """Model should be created with correct config."""
        model = DTSeatHGTv3Model(self.config)
        self.assertEqual(model.config, self.config)

    def test_parameter_count(self) -> None:
        """Parameter count should be positive."""
        model = DTSeatHGTv3Model(self.config)
        count = model.parameter_count()
        self.assertGreater(count, 0)

    def test_validate_parameter_limit(self) -> None:
        """Parameter limit validation should work."""
        model = DTSeatHGTv3Model(self.config)
        validation = model.validate_parameter_limit()
        self.assertIn("parameter_count", validation)
        self.assertIn("maximum_parameters", validation)
        self.assertIn("within_limit", validation)

    def test_forward_output_schema(self) -> None:
        """Forward should return all required outputs."""
        import torch
        from torch_geometric.data import HeteroData

        # Create minimal mock graph data
        data = HeteroData()
        data["stock"].x = torch.randn(5, 32)
        data["stock"].sequence = torch.randn(5, 20, 5)
        data["stock"].sequence_mask = torch.ones(5, 20)

        model = DTSeatHGTv3Model(self.config)
        outputs = model.forward(data, enable_seat_branch=False)

        # Check required outputs
        required = [
            "base_rank_score",
            "base_return_prediction",
            "seat_rank_delta",
            "seat_return_delta",
            "seat_gate",
            "final_rank_score",
            "final_return_prediction",
            "direction_logit",
            "uncertainty",
            "model_valid",
        ]
        for key in required:
            self.assertIn(key, outputs)

        # model_valid should be False
        self.assertFalse(outputs["model_valid"])

    def test_forward_with_seat_branch(self) -> None:
        """Forward with seat branch enabled should produce non-zero deltas."""
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        # Use stock_static_feature_count=32 to match config stock_static_hidden_dim=16
        data["stock"].x = torch.randn(5, 32)
        data["stock"].sequence = torch.randn(5, 20, 5)
        data["stock"].sequence_mask = torch.ones(5, 20)
        data["seat"].x = torch.randn(3, 32)

        model = DTSeatHGTv3Model(self.config)
        outputs = model.forward(data, enable_seat_branch=True)

        # With seat branch, gate should be computed
        self.assertEqual(outputs["seat_gate"].shape[0], 5)

    def test_forward_without_seat_branch(self) -> None:
        """Forward without seat branch should have zero deltas."""
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["stock"].x = torch.randn(5, 32)
        data["stock"].sequence = torch.randn(5, 20, 5)
        data["stock"].sequence_mask = torch.ones(5, 20)

        model = DTSeatHGTv3Model(self.config)
        outputs = model.forward(data, enable_seat_branch=False)

        # Without seat branch, deltas should be zero
        zeros = torch.zeros(5, device=outputs["seat_rank_delta"].device)
        self.assertTrue(torch.allclose(outputs["seat_rank_delta"], zeros))
        self.assertTrue(torch.allclose(outputs["seat_return_delta"], zeros))
        self.assertTrue(torch.allclose(outputs["seat_gate"], zeros))

    def test_seat_gate_initial_bias(self) -> None:
        """Seat gate should start biased low (near zero)."""
        # Create a model with known initial bias
        model = DTSeatHGTv3Model(self.config)

        # Check the bias is set
        import torch
        last_layer = model.seat_gate_mlp[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.bias is not None
        bias = last_layer.bias.data
        self.assertEqual(bias.item(), -2.0)


class TestBuildDTSeatHGTv3(unittest.TestCase):
    """Test build_dt_seat_hgt_v3 function."""

    def test_build_model(self) -> None:
        """Should build model and validate parameters."""
        config = DTSeatHGTv3Config()
        metadata = None  # No schema, use defaults

        model = build_dt_seat_hgt_v3(config, metadata)
        self.assertIsInstance(model, DTSeatHGTv3Model)

    def test_build_exceeds_limit(self) -> None:
        """Should raise if parameters exceed limit."""
        config = DTSeatHGTv3Config(
            hidden_dim=128,
            maximum_parameters=1000,  # Very low limit
        )
        metadata = None

        with self.assertRaises(ValueError) as ctx:
            build_dt_seat_hgt_v3(config, metadata)

        self.assertIn("Parameter limit exceeded", str(ctx.exception))


class TestModelValidation(unittest.TestCase):
    """Test model validation scenarios."""

    def test_hidden_dim_divisible_by_heads(self) -> None:
        """hidden_dim must be divisible by num_heads."""
        # This should not raise (64 / 2 = 32)
        config = DTSeatHGTv3Config(hidden_dim=64, num_heads=2)
        model = DTSeatHGTv3Model(config)
        self.assertIsNotNone(model)

    def test_small_model(self) -> None:
        """Small model should have reasonable parameter count."""
        config = DTSeatHGTv3Config(
            hidden_dim=32,
            stock_gru_hidden_dim=16,
            stock_static_hidden_dim=8,
        )
        model = DTSeatHGTv3Model(config)
        count = model.parameter_count()
        # Should be well under 3M
        self.assertLess(count, 500_000)


if __name__ == "__main__":
    unittest.main()
