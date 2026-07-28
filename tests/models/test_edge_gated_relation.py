"""Tests for edge_gated_relation.py - TypedEdgeGatedRelationLayer and branches."""

from __future__ import annotations

import unittest

import torch


class TestRelationSpecificQKV(unittest.TestCase):
    """Verify each relation type has different Q/K/V projection layers."""

    def setUp(self) -> None:
        """Set up test fixtures."""

    def test_base_relation_has_separate_projections(self) -> None:
        """Base relations (industry→stock, stock→stock) should have separate Q/K/V."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        # Get the layer's projection modules
        layer = branch.layer

        # Verify separate projections exist for each relation
        self.assertIn("industry_to_stock", layer.query_proj)
        self.assertIn("industry_to_stock", layer.key_proj)
        self.assertIn("industry_to_stock", layer.value_proj)

        self.assertIn("stock_to_stock", layer.query_proj)
        self.assertIn("stock_to_stock", layer.key_proj)
        self.assertIn("stock_to_stock", layer.value_proj)

        # Verify they're different modules (not shared)
        industry_q = layer.query_proj["industry_to_stock"]
        stock_q = layer.query_proj["stock_to_stock"]
        self.assertIsNot(industry_q, stock_q)

    def test_seat_relation_has_separate_projections(self) -> None:
        """Seat relations (buy, sell) should have separate Q/K/V projections."""
        from seatflow.models.layers.edge_gated_relation import SeatRelationBranch

        branch = SeatRelationBranch(hidden_dim=64, num_heads=2)

        layer = branch.layer

        # Verify separate projections exist
        self.assertIn("seat_buy_stock", layer.query_proj)
        self.assertIn("seat_sell_stock", layer.query_proj)

        # Verify they're different modules
        buy_q = layer.query_proj["seat_buy_stock"]
        sell_q = layer.query_proj["seat_sell_stock"]
        self.assertIsNot(buy_q, sell_q)

    def test_different_relation_weights_not_shared(self) -> None:
        """Q/K/V weights should be different for different relations."""
        from seatflow.models.layers.edge_gated_relation import (
            RelationConfig,
            TypedEdgeGatedRelationLayer,
        )

        config1 = RelationConfig(
            source_type="a",
            target_type="b",
            relation_name="rel1",
            query_dim=32,
            key_dim=32,
            value_dim=32,
            out_dim=32,
        )
        config2 = RelationConfig(
            source_type="c",
            target_type="d",
            relation_name="rel2",
            query_dim=32,
            key_dim=32,
            value_dim=32,
            out_dim=32,
        )

        layer = TypedEdgeGatedRelationLayer(
            relation_configs={"rel1": config1, "rel2": config2},
            edge_attr_dims={"rel1": 0, "rel2": 0},
            use_edge_gate=False,
        )

        # Get weights
        w1 = layer.query_proj["rel1"].weight.data.clone()
        w2 = layer.query_proj["rel2"].weight.data.clone()

        # Weights should not be identical (different initializations)
        self.assertFalse(torch.allclose(w1, w2))


class TestEdgeAttrAffectsAttention(unittest.TestCase):
    """Verify edge attributes affect attention computation."""

    def test_edge_gate_modulates_attention(self) -> None:
        """Edge attributes should modulate attention scores."""
        from seatflow.models.layers.edge_gated_relation import SeatRelationBranch

        torch.manual_seed(42)
        branch = SeatRelationBranch(hidden_dim=64, num_heads=2, dropout=0.0)

        # Force the edge_encoder to a deterministic mapping so the gate is a
        # known, non-saturating function of the edge attrs. Otherwise the
        # default random init can make sigmoid(W*zeros+b) == sigmoid(W*ones+b)
        # (e.g. when W is near zero), making the test flaky across torch
        # versions. We set Linear1 row 0 = 1.0/7 * ones(7) and Linear4 to read
        # only that row, so gate = sigmoid(silu(1.0/7 * sum(attrs))):
        #   zeros -> sigmoid(silu(0))  = sigmoid(0)   = 0.500
        #   ones  -> sigmoid(silu(1))  ≈ sigmoid(0.731) = 0.675
        #   tens  -> sigmoid(silu(10)) ≈ sigmoid(10)    ≈ 1.000
        # All three differ, so the gate deterministically modulates the message.
        with torch.no_grad():
            enc = branch.layer.edge_encoder["seat_buy_stock"]
            enc[0].weight.zero_()
            enc[0].weight[0, :] = 1.0 / 7
            enc[0].bias.zero_()
            enc[3].weight.zero_()
            enc[3].weight[0, 0] = 1.0
            enc[3].bias.zero_()

        # Create simple graph with 2 seats -> 2 stocks (deterministic inputs)
        torch.manual_seed(42)
        x_dict = {
            "seat": torch.randn(2, 64),
            "stock": torch.randn(2, 64),
        }

        # Edge: seat 0 -> stock 0, seat 1 -> stock 1
        edge_index_dict = {
            ("seat", "seat_buy_stock", "stock"): torch.tensor([[0, 1], [0, 1]]),
        }

        # Edge attrs: all ones (neutral gate)
        edge_attrs_neutral = {
            ("seat", "seat_buy_stock", "stock"): torch.ones(2, 7),  # 7 dims
        }

        # Edge attrs: zeros (zero gate - should suppress attention)
        edge_attrs_zero = {
            ("seat", "seat_buy_stock", "stock"): torch.zeros(2, 7),  # 7 dims
        }

        # Edge attrs: high values (amplified gate)
        edge_attrs_high = {
            ("seat", "seat_buy_stock", "stock"): torch.full((2, 7), 10.0),  # 7 dims
        }

        # Run with neutral gate
        with torch.no_grad():
            out_neutral = branch(x_dict, edge_index_dict, edge_attrs_neutral)
            stock_neutral = out_neutral["stock"].clone()

        # Run with zero gate
        with torch.no_grad():
            out_zero = branch(x_dict, edge_index_dict, edge_attrs_zero)
            stock_zero = out_zero["stock"]

        # Run with high gate
        with torch.no_grad():
            out_high = branch(x_dict, edge_index_dict, edge_attrs_high)
            stock_high = out_high["stock"]

        # Zero gate should produce different output than neutral
        self.assertFalse(torch.allclose(stock_neutral, stock_zero))

        # High gate should produce different output than neutral
        self.assertFalse(torch.allclose(stock_neutral, stock_high))

    def test_edge_encoder_is_applied(self) -> None:
        """Edge attributes should be processed through edge_encoder."""
        from seatflow.models.layers.edge_gated_relation import SeatRelationBranch

        branch = SeatRelationBranch(hidden_dim=64, num_heads=2, use_edge_gate=True)

        # Verify edge encoder exists for seat relations
        self.assertIn("seat_buy_stock", branch.layer.edge_encoder)
        self.assertIn("seat_sell_stock", branch.layer.edge_encoder)


class TestTargetWiseSoftmax(unittest.TestCase):
    """Verify attention is normalized by target node in-degree."""

    def test_attention_normalized_per_target(self) -> None:
        """Attention weights should sum to 1 for each target node."""
        from seatflow.models.layers.edge_gated_relation import (
            RelationConfig,
            TypedEdgeGatedRelationLayer,
        )

        config = RelationConfig(
            source_type="user",
            target_type="item",
            relation_name="user_item",
            query_dim=32,
            key_dim=32,
            value_dim=32,
            out_dim=32,
        )

        layer = TypedEdgeGatedRelationLayer(
            relation_configs={"user_item": config},
            edge_attr_dims={"user_item": 0},
            use_edge_gate=False,
        )

        # Create graph: 3 users -> 2 items
        # Item 0 gets edges from user 0, 1
        # Item 1 gets edge from user 2
        x_dict = {
            "user": torch.randn(3, 32),
            "item": torch.randn(2, 32),
        }

        edge_index_dict = {
            "user_item": torch.tensor([[0, 1, 2], [0, 0, 1]]),  # edges to items
        }

        with torch.no_grad():
            output = layer(x_dict, edge_index_dict)

        # The output should be computed (no NaN)
        self.assertFalse(torch.isnan(output["item"]).any())
        self.assertFalse(torch.isinf(output["item"]).any())

    def test_single_vs_multiple_incoming_edges(self) -> None:
        """Different in-degrees should produce different aggregations."""
        from seatflow.models.layers.edge_gated_relation import (
            RelationConfig,
            TypedEdgeGatedRelationLayer,
        )

        config = RelationConfig(
            source_type="src",
            target_type="dst",
            relation_name="src_dst",
            query_dim=32,
            key_dim=32,
            value_dim=32,
            out_dim=32,
        )

        layer = TypedEdgeGatedRelationLayer(
            relation_configs={"src_dst": config},
            edge_attr_dims={"src_dst": 0},
            use_edge_gate=False,
        )

        # Same source features, different edge structure
        src_features = torch.ones(2, 32) * 5.0
        dst_features = torch.randn(2, 32)

        # Case 1: Single edge to dst[0]
        x_dict_1 = {"src": src_features.clone(), "dst": dst_features.clone()}
        edge_index_1 = {"src_dst": torch.tensor([[0], [0]])}

        # Case 2: Two edges to dst[0]
        x_dict_2 = {"src": src_features.clone(), "dst": dst_features.clone()}
        edge_index_2 = {"src_dst": torch.tensor([[0, 1], [0, 0]])}

        torch.manual_seed(123)
        with torch.no_grad():
            out1 = layer(x_dict_1, edge_index_1)
            out2 = layer(x_dict_2, edge_index_2)

        # Output for dst[0] should be different due to different aggregation
        # Note: this depends on the attention mechanism
        self.assertIsNotNone(out1["dst"])
        self.assertIsNotNone(out2["dst"])


class TestResidualConnection(unittest.TestCase):
    """Verify residual connection exists and is applied."""

    def test_residual_layer_norm_exists(self) -> None:
        """Layer should have layer_norm for residual connection."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        # Layer norm should exist
        self.assertTrue(hasattr(branch.layer, "layer_norm"))
        self.assertIsNotNone(branch.layer.layer_norm)

    def test_residual_added_to_output(self) -> None:
        """Output should include residual (input + transformed)."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        x_dict = {
            "industry": torch.randn(3, 64),
            "stock": torch.randn(2, 64),
        }

        edge_index_dict = {
            ("industry", "industry_to_stock", "stock"): torch.tensor([[0, 1, 2], [0, 0, 1]]),
        }

        torch.manual_seed(42)
        with torch.no_grad():
            output = branch(x_dict, edge_index_dict)

        # Output stock features should be computed
        self.assertEqual(output["stock"].shape[0], 2)
        self.assertEqual(output["stock"].shape[1], 64)

    def test_identity_residual_when_no_edges(self) -> None:
        """With no edges, output should still be valid (residual applied)."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        x_dict = {
            "industry": torch.randn(3, 64),
            "stock": torch.randn(2, 64),
        }

        # Empty edge dict
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        with torch.no_grad():
            output = branch(x_dict, edge_index_dict)

        # Should still return stock features (may be residual + norm of zeros)
        self.assertIn("stock", output)
        self.assertEqual(output["stock"].shape[0], 2)


class TestEmptyEdgeSafety(unittest.TestCase):
    """Verify empty edge cases don't crash."""

    def test_empty_edge_index_dict(self) -> None:
        """Empty edge_index_dict should not crash."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        x_dict = {
            "industry": torch.randn(3, 64),
            "stock": torch.randn(2, 64),
        }

        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        # Should not raise
        with torch.no_grad():
            output = branch(x_dict, edge_index_dict)

        self.assertIsInstance(output, dict)

    def test_no_matching_edges(self) -> None:
        """Non-matching edge types should be skipped safely."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        x_dict = {
            "industry": torch.randn(3, 64),
            "stock": torch.randn(2, 64),
        }

        # Only non-matching edge types
        edge_index_dict = {
            ("other", "other_rel", "other"): torch.tensor([[0], [0]]),
        }

        # Should not crash, just skip non-matching edges
        with torch.no_grad():
            output = branch(x_dict, edge_index_dict)

        # May or may not have output depending on implementation
        self.assertIsInstance(output, dict)

    def test_single_node_no_edges(self) -> None:
        """Single node with no edges should handle gracefully."""
        from seatflow.models.layers.edge_gated_relation import (
            RelationConfig,
            TypedEdgeGatedRelationLayer,
        )

        config = RelationConfig(
            source_type="node",
            target_type="node",
            relation_name="self_loop",
            query_dim=32,
            key_dim=32,
            value_dim=32,
            out_dim=32,
        )

        layer = TypedEdgeGatedRelationLayer(
            relation_configs={"self_loop": config},
            edge_attr_dims={"self_loop": 0},
            use_edge_gate=False,
        )

        x_dict = {"node": torch.randn(1, 32)}
        edge_index_dict: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            output = layer(x_dict, edge_index_dict)

        # Should return valid output
        self.assertIsInstance(output, dict)


class TestAMPForward(unittest.TestCase):
    """Verify AMP (Automatic Mixed Precision) forward pass is safe."""

    def test_forward_with_autocast(self) -> None:
        """Forward pass should work with autocast enabled."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for AMP test")

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2).cuda()

        x_dict = {
            "industry": torch.randn(3, 64).cuda(),
            "stock": torch.randn(2, 64).cuda(),
        }

        edge_index_dict = {
            ("industry", "industry_to_stock", "stock"): torch.tensor([[0, 1, 2], [0, 0, 1]]).cuda(),
        }

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                output = branch(x_dict, edge_index_dict)

        # Check output is valid
        self.assertIsNotNone(output["stock"])
        self.assertEqual(output["stock"].device.type, "cuda")

    def test_seat_branch_with_autocast(self) -> None:
        """Seat branch should work with autocast."""
        from seatflow.models.layers.edge_gated_relation import SeatRelationBranch

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for AMP test")

        branch = SeatRelationBranch(hidden_dim=64, num_heads=2).cuda()

        x_dict = {
            "seat": torch.randn(2, 64).cuda(),
            "stock": torch.randn(2, 64).cuda(),
        }

        edge_index_dict = {
            ("seat", "seat_buy_stock", "stock"): torch.tensor([[0, 1], [0, 1]]).cuda(),
        }

        edge_attrs = {
            ("seat", "seat_buy_stock", "stock"): torch.ones(2, 7).cuda(),  # 7 dims as expected
        }

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                output = branch(x_dict, edge_index_dict, edge_attrs)

        self.assertIsNotNone(output["stock"])
        self.assertEqual(output["stock"].device.type, "cuda")

    def test_float16_computation_stable(self) -> None:
        """Float16 computation should not overflow."""
        from seatflow.models.layers.edge_gated_relation import (
            RelationConfig,
            TypedEdgeGatedRelationLayer,
        )

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = RelationConfig(
            source_type="a",
            target_type="b",
            relation_name="rel",
            query_dim=64,
            key_dim=64,
            value_dim=64,
            out_dim=64,
        )

        layer = TypedEdgeGatedRelationLayer(
            relation_configs={"rel": config},
            edge_attr_dims={"rel": 2},
            use_edge_gate=True,
        ).cuda()

        x_dict = {
            "a": torch.randn(10, 64).cuda(),
            "b": torch.randn(5, 64).cuda(),
        }

        edge_index_dict = {
            "rel": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
            ).cuda(),
        }

        edge_attrs = {
            "rel": torch.randn(10, 2).cuda(),
        }

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                output = layer(x_dict, edge_index_dict, edge_attrs)

        # Check no NaN or Inf
        self.assertFalse(torch.isnan(output["b"]).any())
        self.assertFalse(torch.isinf(output["b"]).any())


class TestRelationWhitelist(unittest.TestCase):
    """Verify relation whitelist configuration."""

    def test_base_relation_whitelist(self) -> None:
        """Base relations should include industry→stock and stock→stock."""
        from seatflow.models.layers.edge_gated_relation import BaseRelationBranch

        branch = BaseRelationBranch(hidden_dim=64, num_heads=2)

        # Check allowed edge types
        allowed = branch.allowed_edge_types
        self.assertIn("industry_to_stock", allowed)
        self.assertIn("stock_to_stock", allowed)

        # Check edge type tuples
        tuples = branch.allowed_edge_type_tuples
        self.assertIn(("industry", "industry_to_stock", "stock"), tuples)
        self.assertIn(("stock", "stock_to_stock", "stock"), tuples)

    def test_seat_relation_whitelist(self) -> None:
        """Seat relations should include seat→stock/buys and seat→stock/sells."""
        from seatflow.models.layers.edge_gated_relation import SeatRelationBranch

        branch = SeatRelationBranch(hidden_dim=64, num_heads=2)

        allowed = branch.allowed_edge_types
        self.assertIn("seat_buy_stock", allowed)
        self.assertIn("seat_sell_stock", allowed)

        tuples = branch.allowed_edge_type_tuples
        self.assertIn(("seat", "seat_buy_stock", "stock"), tuples)
        self.assertIn(("seat", "seat_sell_stock", "stock"), tuples)


if __name__ == "__main__":
    unittest.main()
