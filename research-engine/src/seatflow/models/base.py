"""Base market return model (Task 15).

Predicts raw excess return from market-level features only.
No seat-level or event-level information is used.
This serves as the baseline for the FDG architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BasePrediction:
    """Output of the Base model for a single stock."""

    stock_id: str
    prediction_date: str
    horizon: int
    raw_excess_return: float
    confidence: float = 1.0
    path: str = "Base"


class BaseMarketModel(nn.Module):
    """Simple MLP baseline using only market-level features.

    Architecture:
    - Input: market features (volatility, excess returns, turnover)
    - Hidden: 2-layer MLP with ReLU and dropout
    - Output: scalar excess return prediction per horizon
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_horizons: int = 3,  # h=1,3,5
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Market features [batch, input_dim]

        Returns:
            Predictions [batch, num_horizons] for h=1,3,5
        """
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self.num_horizons


class BaseLoss(nn.Module):
    """Combined Huber + Rank loss for base model."""

    def __init__(self, huber_delta: float = 1.0, rank_weight: float = 0.1) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.rank_weight = rank_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: [batch, horizons]
            target: [batch, horizons]
        """
        huber_loss = self.huber(pred, target)
        rank_loss = self._rank_loss(pred, target)
        return huber_loss + self.rank_weight * rank_loss

    @staticmethod
    def _rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss: pred order should match target order."""
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        # Flatten horizons for rank computation
        p = pred.flatten()
        t = target.flatten()
        # Pairwise differences
        n = p.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        pred_diff = p.unsqueeze(0) - p.unsqueeze(1)  # [n, n]
        target_diff = t.unsqueeze(0) - t.unsqueeze(1)
        # Sign agreement
        sign_agreement = torch.sign(pred_diff) * torch.sign(target_diff)
        # Penalize disagreements
        rank_penalty = torch.clamp(-sign_agreement, min=0.0)
        return rank_penalty.mean()

    def decompose(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """Return decomposed loss components."""
        huber_val = self.huber(pred, target).item()
        rank_val = self._rank_loss(pred, target).item()
        return {
            "huber": huber_val,
            "rank": rank_val,
            "total": huber_val + self.rank_weight * rank_val,
        }
