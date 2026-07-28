"""Seat style encoder for R5 DT-SeatHGT v3.

This module provides a learnable seat style encoder that:
- Uses K learnable style prototypes (registered as nn.Parameter)
- Computes soft style probabilities from historical features
- Includes conditional entropy and marginal balance regularization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SeatStyleEncoderConfig:
    """Configuration for seat style encoder."""

    seat_style_count: int = 5  # K=5 style prototypes
    seat_style_embedding_dim: int = 16
    feature_dim: int = 14  # Number of input features from SeatProfileV2
    hidden_dim: int = 32  # MLP hidden dimension

    def __post_init__(self) -> None:
        if self.seat_style_count <= 0:
            raise ValueError("seat_style_count must be positive")
        if self.seat_style_embedding_dim <= 0:
            raise ValueError("seat_style_embedding_dim must be positive")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

    @property
    def target_usage(self) -> float:
        """Target usage rate per style = 1/K."""
        return 1.0 / self.seat_style_count


class SeatStyleEncoder(nn.Module):
    """Learnable seat style encoder.

    Takes seat profile features and produces:
    - style_probs: [N_seat, K] soft probabilities over K styles
    - style_embedding: [N_seat, D] style prototype weighted embedding
    - style_confidence: [N_seat] confidence in style assignment

    The encoder uses learnable prototypes and encourages:
    - Low conditional entropy: each seat has a relatively clear soft classification
    - High marginal balance: K styles are reasonably used across the batch

    NOTE: Style indices 0-4 are abstract and should NOT be auto-named
    as economic styles (e.g., "chasing", "bottom-fishing") without external validation.
    """

    def __init__(self, config: SeatStyleEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # MLP to process features
        self.mlp = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Style probability head (K classes)
        self.style_head = nn.Linear(config.hidden_dim, config.seat_style_count)

        # Learnable style prototypes (CORRECTED: was prototpyes)
        self.style_prototypes = nn.Parameter(
            torch.randn(config.seat_style_count, config.seat_style_embedding_dim) * 0.1
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: [N_seat, feature_dim] tensor of seat profile features

        Returns:
            dict with:
            - style_probs: [N_seat, K] soft probabilities
            - style_embedding: [N_seat, D] weighted prototype embedding
            - style_confidence: [N_seat] confidence score
            - style_entropy: [N_seat] entropy of style distribution
        """
        # Process features through MLP
        hidden = self.mlp(features)  # [N_seat, hidden_dim]

        # Compute style logits
        style_logits = self.style_head(hidden)  # [N_seat, K]

        # Softmax for probabilities
        style_probs = F.softmax(style_logits, dim=-1)  # [N_seat, K]

        # Compute style embedding as weighted sum of prototypes
        style_embedding = torch.matmul(style_probs, self.style_prototypes)  # [N_seat, D]

        # Confidence as max probability
        style_confidence, _ = style_probs.max(dim=-1)  # [N_seat]

        # Entropy for regularization
        style_entropy = -(style_probs * torch.log(style_probs + 1e-8)).sum(dim=-1)  # [N_seat]

        return {
            "style_probs": style_probs,  # [N_seat, K]
            "style_embedding": style_embedding,  # [N_seat, D]
            "style_confidence": style_confidence,  # [N_seat]
            "style_entropy": style_entropy,  # [N_seat]
        }

    def conditional_entropy_loss(
        self,
        style_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional entropy loss for style diversity.

        LOW conditional entropy = each seat has a clear style (good).
        HIGH conditional entropy = each seat is spread across styles (bad for clarity).

        We want LOW conditional entropy, so we MINIMIZE this loss.
        """
        # Per-sample entropy (conditional entropy)
        entropy = -(style_probs * torch.log(style_probs + 1e-8)).sum(dim=-1)

        # Minimize entropy → maximize confidence
        return entropy.mean()

    def marginal_balance_loss(
        self,
        style_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute marginal balance loss to prevent style collapse.

        HIGH marginal entropy = styles are evenly distributed across batch.
        LOW marginal entropy = one style dominates entire batch.

        Target: each style should be used at ~1/K rate.

        We want HIGH marginal entropy, so we MAXIMIZE mean usage diversity.
        """
        # Mean usage per style across batch
        mean_usage = style_probs.mean(dim=0)  # [K]

        # Target: uniform usage = 1/K
        target_usage = self.config.target_usage

        # Penalize deviation from uniform usage
        return ((mean_usage - target_usage) ** 2).mean()

    def temporal_consistency_loss(
        self,
        style_probs_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute temporal consistency loss for same seat across dates.

        Args:
            style_probs_seq: [N_seat, T, K] style probabilities over time
            valid_mask: [N_seat, T] mask for valid observations (1=valid, 0=missing)

        Returns:
            Consistency loss (MSE between adjacent time steps)
        """
        if style_probs_seq.dim() != 3:
            return torch.tensor(0.0, device=style_probs_seq.device)

        # Differences between adjacent time steps
        diff = style_probs_seq[:, 1:, :] - style_probs_seq[:, :-1, :]  # [N_seat, T-1, K]
        mse = (diff ** 2).mean(dim=(1, 2))  # [N_seat]

        if valid_mask is not None:
            # Count valid transitions
            transitions = valid_mask[:, 1:].float()  # [N_seat, T-1]
            count = transitions.sum(dim=1).clamp(min=1)  # [N_seat]
            mse = (mse * transitions.sum(dim=1)) / count

        return mse.mean()

    def forward_with_losses(
        self,
        features: torch.Tensor,
        style_probs_seq: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with regularization losses.

        Args:
            features: [N_seat, feature_dim] seat features
            style_probs_seq: Optional [N_seat, T, K] for temporal consistency
            valid_mask: Optional [N_seat, T] mask for temporal consistency

        Returns:
            Tuple of (outputs, losses)
        """
        outputs = self.forward(features)

        losses: dict[str, torch.Tensor] = {
            "conditional_entropy": self.conditional_entropy_loss(outputs["style_probs"]),
            "marginal_balance": self.marginal_balance_loss(outputs["style_probs"]),
        }

        if style_probs_seq is not None:
            losses["temporal_consistency"] = self.temporal_consistency_loss(
                style_probs_seq, valid_mask
            )

        return outputs, losses


def build_seat_style_encoder(
    config: SeatStyleEncoderConfig,
) -> SeatStyleEncoder:
    """Build a seat style encoder from config."""
    return SeatStyleEncoder(config)


def validate_style_not_collapsed(
    style_probs: torch.Tensor,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Validate that styles have not collapsed to a single category.

    Args:
        style_probs: [N_seat, K] style probabilities
        threshold: If mean max probability exceeds this, styles may have collapsed

    Returns:
        dict with collapse diagnostic
    """
    max_probs, _ = style_probs.max(dim=-1)  # [N_seat]
    mean_max_prob = max_probs.mean().item()
    std_max_prob = max_probs.std().item()

    return {
        "mean_max_probability": mean_max_prob,
        "std_max_probability": std_max_prob,
        "collapsed": mean_max_prob > threshold,
        "all_near_uniform": std_max_prob < 0.05,
        "diagnostic": (
            "collapsed"
            if mean_max_prob > threshold
            else ("near_uniform" if std_max_prob < 0.05 else "ok")
        ),
    }
