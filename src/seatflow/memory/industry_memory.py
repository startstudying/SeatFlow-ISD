"""Industry memory for temporal graph memory framework.

M11.3 Industry Memory

Industry Memory saves:
- Industry return state
- Industry volatility state
- Industry long/short list activity
- Industry net buy intensity
- Industry seat concentration
- Capital diffusion degree within industry
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class IndustryMemoryConfig:
    """Configuration for industry memory."""

    memory_dim: int = 16
    event_encoder_dim: int = 16
    dropout: float = 0.1


@dataclass
class IndustryMemoryState:
    """Mutable state for industry memory."""

    industry_id: str
    memory: torch.Tensor  # [memory_dim]

    # Return state
    return_state: float = 0.0
    volatility_state: float = 0.0

    # Event activity
    long_activity: float = 0.0  # Total long list activity
    short_activity: float = 0.0  # Total short list activity
    net_buy_intensity: float = 0.0

    # Seat concentration
    seat_concentration: float = 0.0  # HHI of seat participation
    active_seat_count: int = 0

    # Capital diffusion
    capital_diffusion: float = 0.0  # HHI of capital distribution

    # History
    last_update_date: date | None = None
    event_count: int = 0
    memory_available: bool = False


class IndustryMemory(nn.Module):
    """GRU-based industry memory tracking industry-level states.

    Captures:
    - Return momentum and volatility
    - Event activity patterns
    - Seat participation concentration
    - Capital diffusion within industry
    """

    def __init__(self, config: IndustryMemoryConfig) -> None:
        super().__init__()
        self.config = config

        # GRU for memory update
        self.gru_cell = nn.GRUCell(
            input_size=config.event_encoder_dim,
            hidden_size=config.memory_dim,
        )

        # Event encoder
        self.event_encoder = nn.Sequential(
            # long_act, short_act, net, diffusion, seat_hhi, count
            nn.Linear(6, config.event_encoder_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Cold start embedding
        self.cold_start_embedding = nn.Parameter(
            torch.randn(config.memory_dim) * 0.02
        )

    def encode_event(
        self,
        long_activity: float,
        short_activity: float,
        net_buy: float,
        capital_diffusion: float,
        seat_concentration: float,
        active_seat_count: int,
    ) -> torch.Tensor:
        """Encode industry event features.

        Args:
            long_activity: Total long list activity in industry
            short_activity: Total short list activity in industry
            net_buy: Net buy intensity
            capital_diffusion: Capital distribution HHI
            seat_concentration: Seat participation HHI
            active_seat_count: Number of active seats

        Returns:
            [event_encoder_dim] encoded event
        """
        features = torch.tensor([
            long_activity,
            short_activity,
            net_buy,
            capital_diffusion,
            seat_concentration,
            float(active_seat_count) / 100.0,
        ], dtype=torch.float32)
        return cast(torch.Tensor, self.event_encoder(features))

    def initialize_memory(self, industry_id: str) -> IndustryMemoryState:
        """Initialize cold-start memory for a new industry.

        Args:
            industry_id: Unique industry identifier

        Returns:
            IndustryMemoryState with cold-start embedding
        """
        return IndustryMemoryState(
            industry_id=industry_id,
            memory=self.cold_start_embedding.detach().clone(),
            memory_available=False,
        )

    def update_memory(
        self,
        state: IndustryMemoryState,
        event_features: torch.Tensor,
        current_date: date,
        decay_factor: float = 0.95,
    ) -> IndustryMemoryState:
        """Update industry memory with daily event.

        Args:
            state: Current memory state
            event_features: [event_encoder_dim] encoded event
            current_date: Current date
            decay_factor: Decay factor for event states

        Returns:
            Updated IndustryMemoryState
        """
        if not state.memory_available:
            hidden = self.cold_start_embedding.detach().clone()
        else:
            hidden = state.memory

        # GRU update
        new_hidden = self.gru_cell(event_features.unsqueeze(0), hidden.unsqueeze(0))
        new_memory = new_hidden.squeeze(0)

        return IndustryMemoryState(
            industry_id=state.industry_id,
            memory=new_memory,
            return_state=state.return_state,
            volatility_state=state.volatility_state,
            long_activity=state.long_activity * decay_factor,
            short_activity=state.short_activity * decay_factor,
            net_buy_intensity=state.net_buy_intensity * decay_factor,
            seat_concentration=state.seat_concentration * decay_factor,
            active_seat_count=state.active_seat_count,
            capital_diffusion=state.capital_diffusion * decay_factor,
            last_update_date=current_date,
            event_count=state.event_count,
            memory_available=True,
        )

    def apply_return_update(
        self,
        state: IndustryMemoryState,
        industry_return: float,
        industry_volatility: float,
        current_date: date,
    ) -> IndustryMemoryState:
        """Apply return/volatility update.

        Args:
            state: Current memory state
            industry_return: Industry daily return
            industry_volatility: Industry realized volatility
            current_date: Current date

        Returns:
            Updated IndustryMemoryState with return state updated
        """
        return IndustryMemoryState(
            industry_id=state.industry_id,
            memory=state.memory,
            return_state=industry_return,
            volatility_state=industry_volatility,
            long_activity=state.long_activity,
            short_activity=state.short_activity,
            net_buy_intensity=state.net_buy_intensity,
            seat_concentration=state.seat_concentration,
            active_seat_count=state.active_seat_count,
            capital_diffusion=state.capital_diffusion,
            last_update_date=current_date,
            event_count=state.event_count,
            memory_available=state.memory_available,
        )

    def apply_event_update(
        self,
        state: IndustryMemoryState,
        long_activity: float,
        short_activity: float,
        net_buy: float,
        seat_hhi: float,
        seat_count: int,
        capital_hhi: float,
        current_date: date,
    ) -> IndustryMemoryState:
        """Apply event-driven update.

        Args:
            state: Current memory state
            long_activity: Long list activity
            short_activity: Short list activity
            net_buy: Net buy intensity
            seat_hhi: Seat participation HHI
            seat_count: Active seat count
            capital_hhi: Capital distribution HHI
            current_date: Current date

        Returns:
            Updated IndustryMemoryState with event state updated
        """
        return IndustryMemoryState(
            industry_id=state.industry_id,
            memory=state.memory,
            return_state=state.return_state,
            volatility_state=state.volatility_state,
            long_activity=state.long_activity + long_activity,
            short_activity=state.short_activity + short_activity,
            net_buy_intensity=state.net_buy_intensity + net_buy,
            seat_concentration=seat_hhi,
            active_seat_count=seat_count,
            capital_diffusion=capital_hhi,
            last_update_date=current_date,
            event_count=state.event_count + 1,
            memory_available=True,
        )

    def get_memory_features(self, state: IndustryMemoryState) -> torch.Tensor:
        """Get memory features for model input.

        Args:
            state: Current memory state

        Returns:
            [memory_dim + 6] memory features
        """
        metadata = torch.tensor([
            state.return_state,
            state.volatility_state,
            state.long_activity,
            state.short_activity,
            state.net_buy_intensity,
            state.capital_diffusion,
        ])
        return torch.cat([state.memory, metadata])

    def forward(
        self,
        states: dict[str, IndustryMemoryState],
        event_features: dict[str, torch.Tensor],
        current_date: date,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for all industries.

        Args:
            states: Current memory states by industry_id
            event_features: Encoded event features by industry_id
            current_date: Current date

        Returns:
            Memory features by industry_id
        """
        output = {}
        for industry_id, features in event_features.items():
            state = states.get(industry_id)
            if state is None:
                state = self.initialize_memory(industry_id)

            updated_state = self.update_memory(state, features, current_date)
            output[industry_id] = self.get_memory_features(updated_state)

        return output
