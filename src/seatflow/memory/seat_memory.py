"""Seat memory for temporal graph memory framework.

M11.1 Seat Memory

Each seat saves:
- Historical event representation
- Recent buy intensity
- Recent sell intensity
- Net buy state
- Industry preference state
- Stock concentration state
- Continuous activity state
- Event time interval
- Seat soft profile
- Memory last update time

Update:
    m_{s,t} = GRUCell(e_{s,t}, γ(Δt) × m_{s,t-1})

Where:
- e_{s,t}: Daily seat event encoding
- γ(Δt): Memory decay based on trading day interval
- Decay: exp(-λ_m × Δt)

Fixed: memory_half_life = 10 trading days (no search)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SeatMemoryConfig:
    """Configuration for seat memory."""

    memory_dim: int = 32
    event_encoder_dim: int = 32
    memory_half_life_days: int = 10  # Fixed: 10 trading days
    dropout: float = 0.1


@dataclass
class SeatMemoryState:
    """Mutable state for seat memory (not frozen - changes daily)."""

    seat_id: str
    memory: torch.Tensor  # [memory_dim]
    last_update_date: date

    # Aggregated event statistics
    recent_buy_intensity: float = 0.0
    recent_sell_intensity: float = 0.0
    net_buy_state: float = 0.0

    # Behavioral states
    industry_preference: torch.Tensor | None = None  # [industry_count]
    stock_concentration: float = 0.0
    activity_state: float = 0.0  # Continuous activity level

    # Event tracking
    event_count: int = 0
    named_event_count: int = 0
    proxy_event_count: int = 0

    # Memory metadata
    memory_available: bool = True
    event_time_interval: int = 0  # Trading days since last event


@dataclass(frozen=True)
class SeatMemoryConfig_fixed:
    """Fixed configuration constants (no search)."""

    DECAY_LAMBDA: float = 0.0693  # ln(2) / 10 for half-life of 10 days
    MAX_EVENT_AGE: int = 252  # Max event age to track


class SeatMemory(nn.Module):
    """GRU-based seat memory that tracks seat behavior across days.

    Update formula:
        m_{s,t} = GRUCell(e_{s,t}, γ(Δt) × m_{s,t-1})

    Where γ(Δt) = exp(-λ × Δt) is the decay factor.
    """

    def __init__(self, config: SeatMemoryConfig) -> None:
        super().__init__()
        self.config = config

        # GRU cell for memory update
        self.gru_cell = nn.GRUCell(
            input_size=config.event_encoder_dim,
            hidden_size=config.memory_dim,
        )

        # Project aggregated event features to GRU input
        self.event_projection = nn.Sequential(
            # buy_int, sell_int, net, activity, named_ratio, confidence
            nn.Linear(6, config.event_encoder_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Cold start embedding (learnable)
        self.cold_start_embedding = nn.Parameter(
            torch.randn(config.memory_dim) * 0.02
        )

    def compute_decay(
        self,
        last_update_date: date,
        current_date: date,
        trading_days_per_calendar_day: float = 1.0,
    ) -> float:
        """Compute decay factor γ(Δt) = exp(-λ × Δt).

        Args:
            last_update_date: Last date memory was updated
            current_date: Current date
            trading_days_per_calendar_day: Conversion factor

        Returns:
            Decay factor in (0, 1]
        """
        calendar_days = (current_date - last_update_date).days
        trading_days = calendar_days * trading_days_per_calendar_day
        decay = torch.exp(torch.tensor(-SeatMemoryConfig_fixed.DECAY_LAMBDA * trading_days))
        return decay.item()

    def initialize_memory(self, seat_id: str, current_date: date) -> SeatMemoryState:
        """Initialize cold-start memory for a new seat.

        Args:
            seat_id: Unique seat identifier
            current_date: Current date

        Returns:
            SeatMemoryState with cold-start embedding
        """
        return SeatMemoryState(
            seat_id=seat_id,
            memory=self.cold_start_embedding.detach().clone(),
            last_update_date=current_date,
            memory_available=False,  # Not from real events yet
        )

    def update_memory(
        self,
        state: SeatMemoryState,
        daily_event_embedding: torch.Tensor,
        event_stats: dict[str, Any],
        current_date: date,
    ) -> SeatMemoryState:
        """Update seat memory with new event.

        Args:
            state: Current memory state
            daily_event_embedding: [event_encoder_dim] aggregated event embedding
            event_stats: Dict with event statistics
            current_date: Current date

        Returns:
            Updated SeatMemoryState
        """
        if not state.memory_available:
            # First real event - initialize with cold start
            memory = self.cold_start_embedding.detach().clone()
        else:
            # Apply decay to previous memory
            decay = self.compute_decay(state.last_update_date, current_date)
            decayed_memory = state.memory * decay

            # GRU update with event
            memory_input = daily_event_embedding
            gru_input = memory_input.unsqueeze(0)  # [1, event_encoder_dim]

            gru_hidden = decayed_memory.unsqueeze(0)  # [1, memory_dim]
            new_hidden = self.gru_cell(gru_input, gru_hidden)
            memory = new_hidden.squeeze(0)

        # Update statistics
        buy_intensity = event_stats.get("buy_intensity", 0.0)
        sell_intensity = event_stats.get("sell_intensity", 0.0)

        new_event_count = state.event_count + event_stats.get("event_count", 1)
        new_named = state.named_event_count + event_stats.get("named_count", 0)
        new_proxy = state.proxy_event_count + event_stats.get("proxy_count", 0)

        # Decay and update intensity states
        decay = self.compute_decay(state.last_update_date, current_date)
        new_buy_intensity = state.recent_buy_intensity * decay + buy_intensity
        new_sell_intensity = state.recent_sell_intensity * decay + sell_intensity
        new_net_buy = new_buy_intensity - new_sell_intensity
        new_activity = (new_buy_intensity + new_sell_intensity) / max(1, new_event_count)

        # Update concentration (Herfindahl index of stock distribution)
        concentration = event_stats.get("stock_concentration", 0.0)
        new_concentration = state.stock_concentration * decay + concentration

        # Time interval since last event
        interval = max(0, (current_date - state.last_update_date).days)

        return SeatMemoryState(
            seat_id=state.seat_id,
            memory=memory,
            last_update_date=current_date,
            recent_buy_intensity=new_buy_intensity,
            recent_sell_intensity=new_sell_intensity,
            net_buy_state=new_net_buy,
            industry_preference=state.industry_preference,
            stock_concentration=new_concentration,
            activity_state=new_activity,
            event_count=new_event_count,
            named_event_count=new_named,
            proxy_event_count=new_proxy,
            memory_available=True,
            event_time_interval=interval,
        )

    def get_memory_features(
        self,
        state: SeatMemoryState,
        num_industries: int,
    ) -> torch.Tensor:
        """Get memory features for model input.

        Args:
            state: Current memory state
            num_industries: Number of industry categories

        Returns:
            [memory_dim + 4] memory features (memory + metadata)
        """
        # Base memory
        features = [state.memory]

        # Metadata features
        features.append(torch.tensor([
            state.recent_buy_intensity,
            state.recent_sell_intensity,
            state.net_buy_state,
            state.activity_state,
            float(state.memory_available),
            float(state.event_count),
        ]))

        return torch.cat(features)

    def forward(
        self,
        states: dict[str, SeatMemoryState],
        event_embeddings: dict[str, torch.Tensor],
        event_stats: dict[str, dict[str, Any]],
        current_date: date,
        num_industries: int,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for all seats.

        Args:
            states: Current memory states by seat_id
            event_embeddings: Daily event embeddings by seat_id
            event_stats: Event statistics by seat_id
            current_date: Current date
            num_industries: Number of industries

        Returns:
            Memory features by seat_id
        """
        output = {}
        for seat_id, embedding in event_embeddings.items():
            state = states.get(seat_id)
            if state is None:
                state = self.initialize_memory(seat_id, current_date)

            stats = event_stats.get(seat_id, {})

            if embedding.dim() == 2:
                # Multiple events - aggregate
                embedding = embedding.mean(dim=0)

            updated_state = self.update_memory(state, embedding, stats, current_date)
            output[seat_id] = self.get_memory_features(updated_state, num_industries)

        return output
