"""Tests for Spillover model and Relation State Gate (Task 17)."""

import torch
import pytest

from seatflow.models.spillover import (
    RelationStateGate,
    GatedMessagePassing,
    SpilloverModel,
)


class TestRelationStateGate:
    def test_output_shapes(self):
        gate = RelationStateGate(edge_feature_dim=8, hidden_dim=32, regime_dim=4)
        edge_feat = torch.randn(10, 8)
        age = torch.randn(10, 1)
        regime = torch.randn(10, 4)
        state = gate(edge_feat, age, regime)
        assert state["activation"].shape == (10, 1)
        assert state["strength"].shape == (10, 1)
        assert state["direction"].shape == (10, 1)
        assert state["reliability"].shape == (10, 1)

    def test_activation_in_01(self):
        gate = RelationStateGate(edge_feature_dim=8, hidden_dim=32, regime_dim=4)
        edge_feat = torch.randn(20, 8)
        age = torch.randn(20, 1)
        regime = torch.randn(20, 4)
        state = gate(edge_feat, age, regime)
        assert (state["activation"] >= 0).all()
        assert (state["activation"] <= 1).all()

    def test_strength_non_negative(self):
        gate = RelationStateGate(edge_feature_dim=8, hidden_dim=32, regime_dim=4)
        edge_feat = torch.randn(20, 8)
        age = torch.randn(20, 1)
        regime = torch.randn(20, 4)
        state = gate(edge_feat, age, regime)
        assert (state["strength"] >= 0).all()

    def test_direction_in_minus_1_to_1(self):
        gate = RelationStateGate(edge_feature_dim=8, hidden_dim=32, regime_dim=4)
        edge_feat = torch.randn(20, 8)
        age = torch.randn(20, 1)
        regime = torch.randn(20, 4)
        state = gate(edge_feat, age, regime)
        assert (state["direction"] >= -1).all()
        assert (state["direction"] <= 1).all()

    def test_reliability_in_01(self):
        gate = RelationStateGate(edge_feature_dim=8, hidden_dim=32, regime_dim=4)
        edge_feat = torch.randn(20, 8)
        age = torch.randn(20, 1)
        regime = torch.randn(20, 4)
        state = gate(edge_feat, age, regime)
        assert (state["reliability"] >= 0).all()
        assert (state["reliability"] <= 1).all()


class TestGatedMessagePassing:
    def test_zero_activation_zero_message(self):
        """When gate activation=0, message must be exactly 0."""
        mp = GatedMessagePassing(
            source_dim=32, target_dim=32, edge_feature_dim=8, hidden_dim=64, regime_dim=4
        )
        # Create edges
        source_emb = torch.randn(5, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        age = torch.zeros(3, 1)
        regime = torch.zeros(3, 4)

        # Force activation to 0 by setting weights
        with torch.no_grad():
            mp.gate.activation_head.weight.zero_()
            mp.gate.activation_head.bias.fill_(-100)  # sigmoid(-100) ≈ 0

        out = mp(source_emb, edge_feat, edge_index, age, regime)
        # Messages should be near-zero
        assert out.abs().max().item() < 1e-3

    def test_nonzero_gate_changes_output(self):
        """With gate=1 and non-zero params, output should change."""
        mp = GatedMessagePassing(
            source_dim=32, target_dim=32, edge_feature_dim=8, hidden_dim=64, regime_dim=4
        )
        source_emb = torch.randn(5, 32)
        edge_feat = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 0]])
        age = torch.randn(3, 1)
        regime = torch.randn(3, 4)

        out = mp(source_emb, edge_feat, edge_index, age, regime)
        assert out.shape[0] > 0


class TestSpilloverModel:
    def test_forward_shape(self):
        model = SpilloverModel(
            stock_embed_dim=32,
            edge_feature_dim=8,
            hidden_dim=64,
            regime_dim=4,
            num_horizons=3,
        )
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        age = torch.randn(5, 1)
        regime = torch.randn(5, 4)
        out = model(stock_emb, edge_feat, edge_index, age, regime)
        assert out.shape == (10, 3)

    def test_no_edges(self):
        """When there are no edges, spillover should be zeros."""
        model = SpilloverModel(
            stock_embed_dim=32, edge_feature_dim=8, num_horizons=3
        )
        stock_emb = torch.randn(10, 32)
        edge_feat = torch.randn(0, 8)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        age = torch.randn(0, 1)
        regime = torch.randn(0, 4)
        out = model(stock_emb, edge_feat, edge_index, age, regime)
        assert out.shape == (10, 3)
        assert out.abs().max().item() < 1e-5

    def test_output_dim(self):
        model = SpilloverModel(num_horizons=3)
        assert model.output_dim == 3
