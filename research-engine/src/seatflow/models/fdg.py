"""SeatFlow FDG: full model with additive decomposition (Task 17).

Combines Base + Direct + Spillover into a single model with:
- Additive path decomposition: final = base + direct + spillover
- Multiple heads: R1, R3, R5, direction, uncertainty
- Loss: combined return + rank + direction + calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from seatflow.models.base import BaseMarketModel, cumulative_from_daily_path
from seatflow.models.seat import DirectEffectModel
from seatflow.models.spillover import SpilloverModel


@dataclass(frozen=True)
class FDGPredictionBatch:
    """Batch of FDG predictions.

    Contains separate paths for Base, Direct, Spillover, and Final.
    """

    base: torch.Tensor  # [batch, num_horizons]
    base_daily_path: torch.Tensor  # [batch, 5]
    direct: torch.Tensor  # [batch, num_horizons]
    spillover: torch.Tensor  # [batch, num_horizons]
    final: torch.Tensor  # [batch, num_horizons]
    direction: torch.Tensor  # [batch, num_horizons] - tanh
    uncertainty: torch.Tensor  # [batch, num_horizons] - softplus
    direct_mask: Optional[torch.Tensor] = None  # [batch] bool
    spillover_mask: Optional[torch.Tensor] = None  # [batch] bool
    tradable_mask: Optional[torch.Tensor] = None  # [batch] bool
    base_compatibility: str = "NON_FORMAL_FDG_LEGACY_CUMULATIVE_1_3_5"


class SeatFlowFDG(nn.Module):
    """Full SeatFlow FDG model with additive decomposition.

    final = base + direct + spillover
    """

    def __init__(
        self,
        market_input_dim: int = 16,
        seat_input_dim: int = 8,
        event_input_dim: int = 6,
        edge_feature_dim: int = 8,
        stock_embed_dim: int = 32,
        hidden_dim: int = 64,
        num_horizons: int = 3,
        regime_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_horizons != 3:
            raise ValueError("FDG legacy FDG compatibility requires num_horizons == 3")
        self.num_horizons = num_horizons

        self.base_model = BaseMarketModel.for_nonformal_legacy(
            input_dim=market_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.direct_model = DirectEffectModel(
            seat_input_dim=seat_input_dim,
            event_input_dim=event_input_dim,
            market_input_dim=market_input_dim,
            hidden_dim=hidden_dim,
            seat_embed_dim=stock_embed_dim,
            event_embed_dim=stock_embed_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )
        self.spillover_model = SpilloverModel(
            stock_embed_dim=stock_embed_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            regime_dim=regime_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )

        # Direction and uncertainty heads
        combined_dim = num_horizons * 3  # base + direct + spillover
        self.direction_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_horizons),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(
        self,
        market_features: torch.Tensor,
        seat_features: torch.Tensor,
        event_features: torch.Tensor,
        stock_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        age: torch.Tensor,
        market_regime: torch.Tensor,
        direct_mask: Optional[torch.Tensor] = None,
        spillover_mask: Optional[torch.Tensor] = None,
        seat_mask: Optional[torch.Tensor] = None,
        tradable_mask: Optional[torch.Tensor] = None,
    ) -> FDGPredictionBatch:
        """Forward pass through full FDG model.

        Args:
            market_features: [batch, market_input_dim]
            seat_features: [batch, num_seats, seat_input_dim]
            event_features: [batch, event_input_dim]
            stock_embeddings: [num_stocks, stock_embed_dim]
            edge_features: [num_edges, edge_feature_dim]
            edge_index: [2, num_edges]
            age: [num_edges, 1]
            market_regime: [num_edges, regime_dim]
            direct_mask: [batch] bool - whether stock is in Direct pool
            spillover_mask: [batch] bool - whether stock is in Spillover pool
            seat_mask: [batch, num_seats] bool
            tradable_mask: [batch] bool

        Returns:
            FDGPredictionBatch with all paths
        """
        batch_size = market_features.shape[0]

        # Base is formally a five-day daily path. FDG is not yet refactored, so
        # it uses a non-formal compatibility cumulative 1/3/5 view.
        base_daily_path = self.base_model._forward_nonformal_legacy(market_features)  # [batch, 5]
        base = cumulative_from_daily_path(base_daily_path)  # [batch, num_horizons]

        # Direct prediction (delta)
        direct_delta = self.direct_model(
            seat_features, event_features, market_features, seat_mask
        )

        # Spillover prediction
        if edge_index.shape[1] > 0 and stock_embeddings.shape[0] > 0:
            spillover_delta = self.spillover_model(
                stock_embeddings, edge_features, edge_index, age, market_regime
            )
            # Take first batch_size rows
            if spillover_delta.shape[0] >= batch_size:
                spillover_delta = spillover_delta[:batch_size]
            else:
                # Pad with zeros
                pad = torch.zeros(
                    batch_size - spillover_delta.shape[0],
                    self.num_horizons,
                    device=spillover_delta.device,
                )
                spillover_delta = torch.cat([spillover_delta, pad], dim=0)
        else:
            spillover_delta = torch.zeros(
                batch_size, self.num_horizons, device=market_features.device
            )

        # Additive decomposition
        final = base + direct_delta + spillover_delta

        # Direction and uncertainty
        combined = torch.cat([base, direct_delta, spillover_delta], dim=-1)
        direction = torch.tanh(self.direction_head(combined))
        uncertainty = F.softplus(self.uncertainty_head(combined))

        return FDGPredictionBatch(
            base=base,
            base_daily_path=base_daily_path,
            direct=direct_delta,
            spillover=spillover_delta,
            final=final,
            direction=direction,
            uncertainty=uncertainty,
            direct_mask=direct_mask,
            spillover_mask=spillover_mask,
            tradable_mask=tradable_mask,
        )


class SeatFlowLoss(nn.Module):
    """Combined loss for SeatFlow FDG.

    Components:
    - return_loss: Huber on final prediction
    - rank_loss: pairwise rank consistency
    - direction_loss: BCE on direction sign
    - calibration_loss: uncertainty calibration
    - direct_loss: Huber on direct delta (masked)
    - spillover_loss: Huber on spillover delta (masked)
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        rank_weight: float = 0.1,
        direction_weight: float = 0.2,
        calibration_weight: float = 0.1,
        direct_weight: float = 0.3,
        spillover_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.rank_weight = rank_weight
        self.direction_weight = direction_weight
        self.calibration_weight = calibration_weight
        self.direct_weight = direct_weight
        self.spillover_weight = spillover_weight

    def forward(
        self,
        pred: FDGPredictionBatch,
        target: torch.Tensor,
        direction_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: FDGPredictionBatch
            target: [batch, num_horizons] return targets
            direction_target: [batch, num_horizons] -1/0/1 direction targets
        """
        # Return loss on final
        return_loss = self.huber(pred.final, target)

        # Rank loss
        rank_loss = self._rank_loss(pred.final, target)

        # Direction loss
        dir_loss = torch.tensor(0.0, device=target.device)
        if direction_target is not None:
            dir_pred = (pred.direction + 1) / 2  # to [0, 1]
            dir_tgt = (direction_target + 1) / 2
            dir_loss = F.binary_cross_entropy(dir_pred.clamp(1e-6, 1 - 1e-6), dir_tgt)

        # Calibration loss: penalize overconfidence
        cal_loss = F.mse_loss(pred.uncertainty, (pred.final - target).abs())

        # Direct loss (masked)
        direct_loss = torch.tensor(0.0, device=target.device)
        if pred.direct_mask is not None and pred.direct_mask.any():
            mask = pred.direct_mask
            direct_loss = self.huber(pred.direct[mask], target[mask])

        # Spillover loss (masked)
        spillover_loss = torch.tensor(0.0, device=target.device)
        if pred.spillover_mask is not None and pred.spillover_mask.any():
            mask = pred.spillover_mask
            spillover_loss = self.huber(pred.spillover[mask], target[mask])

        total = (
            return_loss
            + self.rank_weight * rank_loss
            + self.direction_weight * dir_loss
            + self.calibration_weight * cal_loss
            + self.direct_weight * direct_loss
            + self.spillover_weight * spillover_loss
        )

        return cast(torch.Tensor, total)

    @staticmethod
    def _rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)
        p = pred.flatten()
        t = target.flatten()
        pred_diff = p.unsqueeze(0) - p.unsqueeze(1)
        target_diff = t.unsqueeze(0) - t.unsqueeze(1)
        sign_agreement = torch.sign(pred_diff) * torch.sign(target_diff)
        return torch.clamp(-sign_agreement, min=0.0).mean()

    def decompose(
        self,
        pred: FDGPredictionBatch,
        target: torch.Tensor,
        direction_target: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Return decomposed loss components for logging."""
        return_loss = self.huber(pred.final, target).item()
        rank_loss = self._rank_loss(pred.final, target).item()
        dir_loss = 0.0
        if direction_target is not None:
            dir_pred = (pred.direction + 1) / 2
            dir_tgt = (direction_target + 1) / 2
            dir_loss = F.binary_cross_entropy(
                dir_pred.clamp(1e-6, 1 - 1e-6), dir_tgt
            ).item()
        cal_loss = F.mse_loss(pred.uncertainty, (pred.final - target).abs()).item()
        return {
            "return": return_loss,
            "rank": rank_loss,
            "direction": dir_loss,
            "calibration": cal_loss,
            "total": (
                return_loss
                + self.rank_weight * rank_loss
                + self.direction_weight * dir_loss
                + self.calibration_weight * cal_loss
            ),
        }
