"""Event and graph-level models (Task 16 continued).

EventModel handles event-level encoding for the SeatFlow architecture.
This is the event representation used by both Direct and Spillover models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EventPrediction:
    """Event model prediction output."""

    stock_id: str
    event_id: str
    direct_effect: float
    event_embedding: list[float]


class EventModel(nn.Module):
    """Encodes LHB event into a context vector.

    Uses:
    - Event static features (turnover, net amount, reason codes)
    - Temporal features (days since last event)
    - Seat aggregation (pre-computed by SeatEncoder)
    """

    def __init__(
        self,
        event_input_dim: int = 6,
        seat_embed_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Event feature encoder
        self.event_encoder = nn.Sequential(
            nn.Linear(event_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: event attends to seat embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        event_features: torch.Tensor,
        seat_embeddings: torch.Tensor,
        seat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode event with seat context.

        Args:
            event_features: [batch, event_input_dim]
            seat_embeddings: [batch, num_seats, seat_embed_dim]
            seat_mask: [batch, num_seats] bool mask

        Returns:
            Event context embedding [batch, output_dim]
        """
        # Project event features to hidden dim
        event_h = self.event_encoder(event_features)  # [B, hidden]
        event_h = event_h.unsqueeze(1)  # [B, 1, hidden] - as query

        # Project seat embeddings to hidden dim if needed
        if seat_embeddings.shape[-1] != event_h.shape[-1]:
            seat_proj = nn.Linear(
                seat_embeddings.shape[-1], event_h.shape[-1]
            ).to(seat_embeddings.device)
            seat_h = seat_proj(seat_embeddings)
        else:
            seat_h = seat_embeddings

        # Cross-attention: event (query) attends to seats (key/value)
        key_padding_mask = ~seat_mask if seat_mask is not None else None
        attn_out, _ = self.cross_attn(
            event_h, seat_h, seat_h, key_padding_mask=key_padding_mask
        )

        # Squeeze and project
        return self.output_proj(attn_out.squeeze(1))
