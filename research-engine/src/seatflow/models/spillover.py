"""Spillover model and Relation State Gate (Task 17).

The Spillover model predicts indirect return effects on stocks that are
NOT directly on the LHB but are connected via propagation relations.

Relation State Gate: a learnable per-edge gate that controls
- edge activation
- message weight
- Top-K neighbor selection
- positive/negative direction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationStateGate(nn.Module):
    """Per-edge learnable state that gates message passing.

    For each edge, produces:
    - activation: sigmoid in [0, 1] -> whether edge is active
    - strength: softplus >= 0 -> magnitude of message
    - direction: tanh in [-1, 1] -> positive/negative sign
    - reliability: sigmoid in [0, 1] -> trust in edge

    When activation = 0, the message MUST be exactly 0.
    """

    def __init__(
        self,
        edge_feature_dim: int = 8,
        hidden_dim: int = 32,
        regime_dim: int = 4,
    ) -> None:
        super().__init__()
        input_dim = edge_feature_dim + 1 + regime_dim  # +1 for age

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Three output heads
        self.activation_head = nn.Linear(hidden_dim, 1)
        self.strength_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 1)
        self.reliability_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        edge_features: torch.Tensor,
        age: torch.Tensor,
        market_regime: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute relation state for each edge.

        Args:
            edge_features: [num_edges, edge_feature_dim]
            age: [num_edges, 1] - days since edge was created
            market_regime: [num_edges, regime_dim] - current market state

        Returns:
            Dict with keys: activation, strength, direction, reliability
        """
        h = self.net(torch.cat([edge_features, age, market_regime], dim=-1))
        activation = torch.sigmoid(self.activation_head(h))
        strength = F.softplus(self.strength_head(h))
        direction = torch.tanh(self.direction_head(h))
        reliability = torch.sigmoid(self.reliability_head(h))

        return {
            "activation": activation,  # [num_edges, 1]
            "strength": strength,
            "direction": direction,
            "reliability": reliability,
        }


class GatedMessagePassing(nn.Module):
    """Message passing on a heterogeneous graph with relation state gating.

    For each edge type:
    1. Compute RelationStateGate(state)
    2. Filter out edges with activation = 0
    3. Aggregate source embeddings using gated messages
    4. Update target node embeddings
    """

    def __init__(
        self,
        source_dim: int = 32,
        target_dim: int = 32,
        edge_feature_dim: int = 8,
        hidden_dim: int = 64,
        regime_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = RelationStateGate(
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            regime_dim=regime_dim,
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(source_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        age: torch.Tensor,
        market_regime: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate messages from source to target.

        Args:
            source_embeddings: [num_source_nodes, source_dim]
            edge_features: [num_edges, edge_feature_dim]
            edge_index: [2, num_edges] - (source_idx, target_idx)
            age: [num_edges, 1]
            market_regime: [num_edges, regime_dim]

        Returns:
            Updated target embeddings [num_target_nodes, target_dim]
        """
        if edge_index.shape[1] == 0:
            # No edges: return zeros matching number of source nodes
            return torch.zeros(
                source_embeddings.shape[0],
                self.message_mlp[-1].out_features,
                device=source_embeddings.device,
            )

        # Get source embeddings for each edge
        src_idx = edge_index[0]
        src_embs = source_embeddings[src_idx]  # [num_edges, source_dim]

        # Compute messages
        msg_input = torch.cat([src_embs, edge_features], dim=-1)
        messages = self.message_mlp(msg_input)  # [num_edges, target_dim]

        # Apply gate
        state = self.gate(edge_features, age, market_regime)
        gate_value = (
            state["activation"]
            * state["strength"]
            * state["direction"]
            * state["reliability"]
        )
        gated_messages = messages * gate_value  # [num_edges, target_dim]

        # Aggregate at targets
        tgt_idx = edge_index[1]
        num_targets = int(tgt_idx.max().item()) + 1 if tgt_idx.numel() > 0 else 0
        if num_targets == 0:
            return torch.zeros(
                0, gated_messages.shape[-1], device=gated_messages.device
            )
        aggregated = torch.zeros(
            num_targets, gated_messages.shape[-1], device=gated_messages.device
        )
        aggregated.index_add_(0, tgt_idx, gated_messages)

        return aggregated


class SpilloverModel(nn.Module):
    """Predicts indirect spillover effect on non-LHB stocks.

    Uses GatedMessagePassing to aggregate propagation signals from
    LHB-listed stocks through heterogeneous relations.
    """

    def __init__(
        self,
        stock_embed_dim: int = 32,
        edge_feature_dim: int = 8,
        hidden_dim: int = 64,
        regime_dim: int = 4,
        num_horizons: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons

        self.gated_passing = GatedMessagePassing(
            source_dim=stock_embed_dim,
            target_dim=stock_embed_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            regime_dim=regime_dim,
            dropout=dropout,
        )

        # Spillover head: aggregate + project to horizon predictions
        self.spillover_head = nn.Sequential(
            nn.Linear(stock_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(
        self,
        stock_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        age: torch.Tensor,
        market_regime: torch.Tensor,
    ) -> torch.Tensor:
        """Predict spillover effect.

        Args:
            stock_embeddings: [num_stocks, stock_embed_dim]
            edge_features: [num_edges, edge_feature_dim]
            edge_index: [2, num_edges]
            age: [num_edges, 1]
            market_regime: [num_edges, regime_dim]

        Returns:
            Spillover effect [num_stocks, num_horizons]
        """
        if edge_index.shape[1] == 0:
            # No edges: spillover contribution is exactly zero
            return torch.zeros(
                stock_embeddings.shape[0],
                self.num_horizons,
                device=stock_embeddings.device,
            )
        updated = self.gated_passing(
            stock_embeddings, edge_features, edge_index, age, market_regime
        )
        if updated.shape[0] == 0:
            return torch.zeros(
                stock_embeddings.shape[0],
                self.num_horizons,
                device=stock_embeddings.device,
            )
        return self.spillover_head(updated)

    @property
    def output_dim(self) -> int:
        return self.num_horizons
