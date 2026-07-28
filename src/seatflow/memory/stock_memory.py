"""Stock memory for temporal graph memory framework.

M11.2 Stock Memory

Stock Memory saves:
- Historical price-volume state
- Recent long/short list event intensity
- Persistent net buy state
- Seat concentration change
- Industry relative strength
- Seat message history
- Prediction uncertainty history
- Event duration

Update input:
- Daily stock temporal representation
- Seat→Stock daily message
- Industry→Stock daily message
- Stock→Stock daily message

Update:
    m_{i,t}^{stock} = GRUCell([h_{i,t}^{price}; h_{i,t}^{seat}; h_{i,t}^{industry}],
                             m_{i,t-1}^{stock})

FORBIDDEN inputs:
- Yesterday's prediction rank
- Yesterday's Top-5 selection
- Yesterday's portfolio status

Model memory and portfolio strategy MUST be separated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class StockMemoryConfig:
    """Configuration for stock memory."""

    memory_dim: int = 32
    price_encoder_dim: int = 16
    seat_message_dim: int = 16
    industry_message_dim: int = 8
    dropout: float = 0.1


@dataclass
class StockMemoryState:
    """Mutable state for stock memory."""

    stock_id: str
    memory: torch.Tensor  # [memory_dim]

    # Price-volume state
    price_state: torch.Tensor  # [price_encoder_dim]
    volume_state: float = 0.0

    # Event state
    buy_intensity: float = 0.0
    sell_intensity: float = 0.0
    net_buy_state: float = 0.0
    event_duration: int = 0  # Days since first event

    # Seat concentration
    seat_concentration: float = 0.0  # HHI of seat activity
    seat_count: int = 0

    # Industry relative strength
    industry_relative_strength: float = 0.0

    # History tracking
    last_update_date: date | None = None
    event_count: int = 0
    memory_available: bool = False


class StockMemory(nn.Module):
    """GRU-based stock memory tracking price-volume and event states.

    Memory update is strictly decoupled from prediction outputs.
    FORBIDDEN to use prediction rank, Top-5 selection, or portfolio status.
    """

    def __init__(self, config: StockMemoryConfig) -> None:
        super().__init__()
        self.config = config

        # GRU for memory update
        gru_input_dim = (
            config.price_encoder_dim
            + config.seat_message_dim
            + config.industry_message_dim
        )
        self.gru_cell = nn.GRUCell(
            input_size=gru_input_dim,
            hidden_size=config.memory_dim,
        )

        # Price encoder (encodes daily price-volume state)
        self.price_encoder = nn.Sequential(
            nn.Linear(5, config.price_encoder_dim),  # return, volume_ratio, volatility, etc.
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Seat message encoder
        self.seat_message_encoder = nn.Sequential(
            nn.Linear(4, config.seat_message_dim),  # buy_int, sell_int, concentration, count
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Industry message encoder
        self.industry_message_encoder = nn.Sequential(
            nn.Linear(3, config.industry_message_dim),  # return, relative_strength, activity
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Cold start embedding
        self.cold_start_embedding = nn.Parameter(
            torch.randn(config.memory_dim) * 0.02
        )

    def encode_price_state(
        self,
        daily_return: float,
        volume_ratio: float,
        volatility: float,
        bid_ask_spread: float,
        price_momentum: float,
    ) -> torch.Tensor:
        """Encode daily price-volume features.

        Args:
            daily_return: Daily return
            volume_ratio: Volume / average volume
            volatility: Realized volatility
            bid_ask_spread: Bid-ask spread
            price_momentum: Short-term momentum

        Returns:
            [price_encoder_dim] encoded state
        """
        features = torch.tensor([
            daily_return,
            volume_ratio,
            volatility,
            bid_ask_spread,
            price_momentum,
        ], dtype=torch.float32)
        return cast(torch.Tensor, self.price_encoder(features))

    def encode_seat_message(
        self,
        buy_intensity: float,
        sell_intensity: float,
        seat_concentration: float,
        active_seat_count: int,
    ) -> torch.Tensor:
        """Encode seat-related message.

        Args:
            buy_intensity: Sum of buy activity
            sell_intensity: Sum of sell activity
            seat_concentration: HHI of seat activity
            active_seat_count: Number of active seats

        Returns:
            [seat_message_dim] encoded message
        """
        features = torch.tensor([
            buy_intensity,
            sell_intensity,
            seat_concentration,
            float(active_seat_count) / 100.0,  # Normalize
        ], dtype=torch.float32)
        return cast(torch.Tensor, self.seat_message_encoder(features))

    def encode_industry_message(
        self,
        industry_return: float,
        relative_strength: float,
        industry_activity: float,
    ) -> torch.Tensor:
        """Encode industry-related message.

        Args:
            industry_return: Industry daily return
            relative_strength: Stock return - industry return
            industry_activity: Industry-wide event activity

        Returns:
            [industry_message_dim] encoded message
        """
        features = torch.tensor([
            industry_return,
            relative_strength,
            industry_activity,
        ], dtype=torch.float32)
        return cast(torch.Tensor, self.industry_message_encoder(features))

    def initialize_memory(self, stock_id: str) -> StockMemoryState:
        """Initialize cold-start memory for a new stock.

        Args:
            stock_id: Unique stock identifier

        Returns:
            StockMemoryState with cold-start embedding
        """
        return StockMemoryState(
            stock_id=stock_id,
            memory=self.cold_start_embedding.detach().clone(),
            price_state=torch.zeros(self.config.price_encoder_dim),
            memory_available=False,
        )

    def update_memory(
        self,
        state: StockMemoryState,
        price_features: torch.Tensor,
        seat_message: torch.Tensor,
        industry_message: torch.Tensor,
        current_date: date,
        decay_factor: float = 0.9,
    ) -> StockMemoryState:
        """Update stock memory with daily inputs.

        Args:
            state: Current memory state
            price_features: [price_encoder_dim] price-volume features
            seat_message: [seat_message_dim] seat activity message
            industry_message: [industry_message_dim] industry message
            current_date: Current date
            decay_factor: Decay for event states

        Returns:
            Updated StockMemoryState
        """
        # Concatenate all messages
        gru_input = torch.cat([price_features, seat_message, industry_message])

        if not state.memory_available:
            # Use cold start
            hidden = self.cold_start_embedding.detach().clone()
        else:
            hidden = state.memory

        # GRU update
        new_hidden = self.gru_cell(gru_input.unsqueeze(0), hidden.unsqueeze(0))
        new_memory = new_hidden.squeeze(0)

        return StockMemoryState(
            stock_id=state.stock_id,
            memory=new_memory,
            price_state=price_features,
            volume_state=state.volume_state,
            buy_intensity=state.buy_intensity * decay_factor,
            sell_intensity=state.sell_intensity * decay_factor,
            net_buy_state=state.net_buy_state * decay_factor,
            event_duration=state.event_duration,
            seat_concentration=state.seat_concentration * decay_factor,
            seat_count=state.seat_count,
            industry_relative_strength=state.industry_relative_strength,
            last_update_date=current_date,
            event_count=state.event_count,
            memory_available=True,
        )

    def apply_event_update(
        self,
        state: StockMemoryState,
        buy_amount: float,
        sell_amount: float,
        seat_hhi: float,
        seat_count: int,
        current_date: date,
    ) -> StockMemoryState:
        """Apply event-driven updates to memory state.

        Args:
            state: Current memory state
            buy_amount: Total buy amount
            sell_amount: Total sell amount
            seat_hhi: Seat concentration (HHI)
            seat_count: Number of active seats
            current_date: Current date

        Returns:
            Updated StockMemoryState with event state updated
        """
        return StockMemoryState(
            stock_id=state.stock_id,
            memory=state.memory,
            price_state=state.price_state,
            volume_state=state.volume_state,
            buy_intensity=state.buy_intensity + buy_amount,
            sell_intensity=state.sell_intensity + sell_amount,
            net_buy_state=state.net_buy_state + buy_amount - sell_amount,
            event_duration=state.event_duration + 1,
            seat_concentration=seat_hhi,
            seat_count=seat_count,
            industry_relative_strength=state.industry_relative_strength,
            last_update_date=current_date,
            event_count=state.event_count + 1,
            memory_available=state.memory_available,
        )

    def get_memory_features(self, state: StockMemoryState) -> torch.Tensor:
        """Get memory features for model input.

        Args:
            state: Current memory state

        Returns:
            [memory_dim + 5] memory features
        """
        metadata = torch.tensor([
            state.buy_intensity,
            state.sell_intensity,
            state.net_buy_state,
            state.seat_concentration,
            state.industry_relative_strength,
        ])
        return torch.cat([state.memory, metadata])

    def forward(
        self,
        states: dict[str, StockMemoryState],
        price_features: dict[str, torch.Tensor],
        seat_messages: dict[str, torch.Tensor],
        industry_messages: dict[str, torch.Tensor],
        current_date: date,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for all stocks.

        Args:
            states: Current memory states by stock_id
            price_features: Daily price features by stock_id
            seat_messages: Seat messages by stock_id
            industry_messages: Industry messages by stock_id
            current_date: Current date

        Returns:
            Memory features by stock_id
        """
        output = {}
        for stock_id in price_features:
            state = states.get(stock_id)
            if state is None:
                state = self.initialize_memory(stock_id)

            p_feat = price_features[stock_id]
            s_msg = seat_messages.get(stock_id, torch.zeros(self.config.seat_message_dim))
            i_msg = industry_messages.get(stock_id, torch.zeros(self.config.industry_message_dim))

            updated_state = self.update_memory(
                state, p_feat, s_msg, i_msg, current_date
            )
            output[stock_id] = self.get_memory_features(updated_state)

        return output
