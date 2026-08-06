"""Seat, Event, and Direct effect models (Task 16).

SeatModel: encodes seat history features.
EventModel: encodes event-level features.
DirectEffectModel: combines Seat and Event encodings to predict
the direct seat-to-stock effect on LHB-listed stocks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SeatEncoder(nn.Module):
    """Encodes seat participation history.

    Input: seat features (buy/sell amounts, counts, mature return stats)
    Output: seat embedding vector
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        seat_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode seat participation features.

        Args:
            seat_features: [batch, num_seats, input_dim]
            mask: [batch, num_seats] bool mask (True = valid)

        Returns:
            Seat embedding [batch, output_dim]
        """
        h = self.projection(seat_features)  # [B, S, hidden]
        if mask is not None:
            key_padding_mask = ~mask  # attn mask: True = ignore
        else:
            key_padding_mask = None
        attn_out, _ = self.attention(
            h, h, h, key_padding_mask=key_padding_mask
        )
        # Pool: mean over seats
        pooled = attn_out.mean(dim=1)  # [B, hidden]
        return self.output_proj(pooled)


class EventEncoder(nn.Module):
    """Encodes event-level features.

    Input: event features (turnover rate, net amount, reason codes)
    Output: event embedding vector
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, event_features: torch.Tensor) -> torch.Tensor:
        """Encode event features.

        Args:
            event_features: [batch, input_dim]

        Returns:
            Event embedding [batch, output_dim]
        """
        return self.net(event_features)


class DirectEffectModel(nn.Module):
    """Predicts the direct seat→stock effect for LHB-listed stocks.

    Combines:
    - SeatEncoder: aggregate seat participation embeddings
    - EventEncoder: event-level context
    - Market base: base model prediction
    - Direct head: maps combined representation to direct effect

    The Direct prediction is additive on top of Base:
        direct_effect = Base + DirectDelta
    """

    def __init__(
        self,
        seat_input_dim: int = 8,
        event_input_dim: int = 6,
        market_input_dim: int = 16,
        hidden_dim: int = 64,
        seat_embed_dim: int = 32,
        event_embed_dim: int = 32,
        num_horizons: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_horizons = num_horizons

        self.seat_encoder = SeatEncoder(
            input_dim=seat_input_dim,
            hidden_dim=hidden_dim,
            output_dim=seat_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            input_dim=event_input_dim,
            hidden_dim=hidden_dim,
            output_dim=event_embed_dim,
            dropout=dropout,
        )
        # Market context projection
        self.market_proj = nn.Linear(market_input_dim, hidden_dim)

        # Direct head: combine seat + event + market → delta
        combined_dim = seat_embed_dim + event_embed_dim + hidden_dim
        self.direct_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(
        self,
        seat_features: torch.Tensor,
        event_features: torch.Tensor,
        market_features: torch.Tensor,
        seat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict direct effect delta.

        Args:
            seat_features: [batch, num_seats, seat_input_dim]
            event_features: [batch, event_input_dim]
            market_features: [batch, market_input_dim]
            seat_mask: [batch, num_seats] bool mask

        Returns:
            Direct effect delta [batch, num_horizons]
        """
        seat_emb = self.seat_encoder(seat_features, mask=seat_mask)
        event_emb = self.event_encoder(event_features)
        market_emb = self.market_proj(market_features)

        combined = torch.cat([seat_emb, event_emb, market_emb], dim=-1)
        return self.direct_head(combined)

    @property
    def output_dim(self) -> int:
        return self.num_horizons
