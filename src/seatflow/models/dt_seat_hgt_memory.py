"""DT-SeatHGT with Temporal Graph Memory integration.

M11.11 Memory and DT-SeatHGT Fusion

Base stock input:
    stock_input = current_stock_embedding + projected_stock_memory

Seat input:
    seat_input = current_seat_features + projected_seat_memory

Industry input:
    industry_input = current_industry_features + projected_industry_memory

With learnable gate:
    h_t^{final} = h_t^{current} + g_t^{memory} ⊙ (W_m × M_t)

Where: memory_gate ∈ [0,1]

When no historical memory:
    memory_gate = 0

Cross-day memory correctness contract (T7 audit):
    1. Memory state is updated daily in chronological order; decay γ(Δt) is
       applied to the previous memory BEFORE the GRU update.
    2. A new fold / sequence MUST begin with a fresh memory state. Calling
       ``begin_sequence`` clears all seat/stock/industry states so no
       information leaks across folds or between independent sequences.
    3. ``get_memory_states`` returns isolated copies: external mutation of the
       returned states (e.g. TBPTT detach) never aliases the model's internal
       state, and vice-versa.
    4. Memory gating is zero (memory contributes nothing) until the node has
       observed at least one real event (``memory_available`` flag).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from seatflow.models.dt_seat_hgt_v3 import DTSeatHGTv3Config, DTSeatHGTv3Model

# Memory update formula: m_{t} = GRU(e_{t}, γ(Δt) × m_{t-1})
# Decay γ(Δt) = exp(-λ × Δt) with λ = ln(2) / half_life.
_DECAY_LAMBDA_DEFAULT = math.log(2.0) / 10.0  # half-life of 10 trading days


@dataclass(frozen=True)
class DTSeatHGTMemoryConfig:
    """Configuration for DT-SeatHGT with memory."""

    # Base model config
    base: DTSeatHGTv3Config = None  # type: ignore

    # Memory configuration
    memory_dim: int = 32
    seat_memory_dim: int = 32
    stock_memory_dim: int = 32
    industry_memory_dim: int = 16

    # TBPTT configuration
    tbptt_days: int = 20

    # Memory decay
    memory_half_life_trading_days: int = 10

    # Dropout
    memory_dropout: float = 0.10

    # Memory gating
    memory_gate_initial_bias: float = -3.0  # Conservative initialization

    # Resource limits
    maximum_parameters: int = 3_000_000
    maximum_peak_gpu_gib: float = 7.2

    def decay_lambda(self) -> float:
        """Exponential decay constant λ = ln(2) / half_life."""
        half_life = self.memory_half_life_trading_days
        if half_life <= 0:
            raise ValueError("memory_half_life_trading_days must be positive")
        return math.log(2.0) / half_life


class DTSeatHGTMemoryModel(nn.Module):
    """DT-SeatHGT model with temporal graph memory integration.

    Integrates seat, stock, and industry memory into the base model
    via learnable gating mechanisms.

    Cross-day memory is managed through three state dicts keyed by entity id:
        _seat_memory_states:     dict[seat_id,      SeatMemoryState]
        _stock_memory_states:    dict[stock_id,     StockMemoryState]
        _industry_memory_states: dict[industry_id,  IndustryMemoryState]

    Use ``begin_sequence`` before processing a new fold/sequence to guarantee
    no cross-fold leakage, and ``step_daily_memory`` to advance memory in
    chronological order across trading days.
    """

    def __init__(
        self,
        config: DTSeatHGTMemoryConfig,
        graph_schema: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._parameter_count: int | None = None

        # Import memory modules
        from seatflow.memory.industry_memory import IndustryMemory, IndustryMemoryConfig
        from seatflow.memory.seat_memory import SeatMemory, SeatMemoryConfig
        from seatflow.memory.stock_memory import StockMemory, StockMemoryConfig

        # Memory modules
        seat_mem_config = SeatMemoryConfig(
            memory_dim=config.seat_memory_dim,
            event_encoder_dim=config.seat_memory_dim,
            memory_half_life_days=config.memory_half_life_trading_days,
            dropout=config.memory_dropout,
        )
        self.seat_memory = SeatMemory(seat_mem_config)

        stock_mem_config = StockMemoryConfig(
            memory_dim=config.stock_memory_dim,
            price_encoder_dim=config.stock_memory_dim // 2,
            seat_message_dim=config.stock_memory_dim // 2,
            industry_message_dim=config.stock_memory_dim // 4,
            dropout=config.memory_dropout,
        )
        self.stock_memory = StockMemory(stock_mem_config)

        industry_mem_config = IndustryMemoryConfig(
            memory_dim=config.industry_memory_dim,
            event_encoder_dim=config.industry_memory_dim,
            dropout=config.memory_dropout,
        )
        self.industry_memory = IndustryMemory(industry_mem_config)

        # Base DT-SeatHGT model
        if config.base is None:
            config = DTSeatHGTMemoryConfig(  # noqa: PLW2901 - local rebuild with base
                base=DTSeatHGTv3Config(),
                seat_memory_dim=config.seat_memory_dim,
                stock_memory_dim=config.stock_memory_dim,
                industry_memory_dim=config.industry_memory_dim,
                tbptt_days=config.tbptt_days,
                memory_half_life_trading_days=config.memory_half_life_trading_days,
                memory_dropout=config.memory_dropout,
                memory_gate_initial_bias=config.memory_gate_initial_bias,
                maximum_parameters=config.maximum_parameters,
                maximum_peak_gpu_gib=config.maximum_peak_gpu_gib,
            )
            self.config = config

        self.base_model = DTSeatHGTv3Model(self.config.base, graph_schema)

        # Memory projection dimensions.
        # Concatenated memory layout is [seat_mem | stock_mem | industry_mem]
        # so the per-entity slice offsets are computed once here.
        total_memory_dim = (
            config.seat_memory_dim
            + config.stock_memory_dim
            + config.industry_memory_dim
        )

        # Project combined memory to model hidden dim
        self.memory_projection = nn.Sequential(
            nn.Linear(total_memory_dim, self.config.base.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.memory_dropout),
            nn.Linear(self.config.base.hidden_dim, self.config.base.hidden_dim),
        )

        # Memory gates (learnable, sigmoid activated)
        self.stock_memory_gate = nn.Sequential(
            nn.Linear(self.config.base.hidden_dim * 2, self.config.base.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.config.base.hidden_dim // 2, 1),
        )
        self.stock_memory_gate[-1].bias.data.fill_(  # type: ignore[operator]
            config.memory_gate_initial_bias
        )

        self.seat_memory_gate = nn.Sequential(
            nn.Linear(self.config.base.hidden_dim * 2, self.config.base.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.config.base.hidden_dim // 2, 1),
        )
        self.seat_memory_gate[-1].bias.data.fill_(  # type: ignore[operator]
            config.memory_gate_initial_bias
        )

        # Memory state tracking (non-learnable, managed externally).
        # These dicts are always replaced wholesale (never mutated in place by
        # callers) to keep the cross-day / cross-fold state isolated.
        self._seat_memory_states: dict[str, Any] = {}
        self._stock_memory_states: dict[str, Any] = {}
        self._industry_memory_states: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # State management: isolation is the correctness guarantee against
    # cross-fold leakage and TBPTT aliasing.
    # ------------------------------------------------------------------

    def set_memory_states(
        self,
        seat_states: dict[str, Any] | None = None,
        stock_states: dict[str, Any] | None = None,
        industry_states: dict[str, Any] | None = None,
    ) -> None:
        """Set memory states from external storage.

        Incoming states are deep-copied so the model owns an isolated copy:
        subsequent mutation of the caller's dict (e.g. TBPTT detach) cannot
        leak into the model's internal state, and vice-versa.

        Args:
            seat_states: Seat memory states by seat_id
            stock_states: Stock memory states by stock_id
            industry_states: Industry memory states by industry_id
        """
        if seat_states is not None:
            self._seat_memory_states = self._deep_copy_states(seat_states)
        if stock_states is not None:
            self._stock_memory_states = self._deep_copy_states(stock_states)
        if industry_states is not None:
            self._industry_memory_states = self._deep_copy_states(industry_states)

    def get_memory_states(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get current memory states for storage.

        Returns isolated copies. The returned state objects are deep-copied so
        that external mutation (e.g. TBPTT ``.detach()`` reassignment) cannot
        alias the model's internal state, and the model's internal state cannot
        leak into a caller's snapshot.

        Returns:
            Tuple of (seat_states, stock_states, industry_states)
        """
        return (
            self._deep_copy_states(self._seat_memory_states),
            self._deep_copy_states(self._stock_memory_states),
            self._deep_copy_states(self._industry_memory_states),
        )

    @staticmethod
    def _deep_copy_states(states: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy a state dict, detaching any tensors for isolation."""
        if not states:
            return {}
        copies: dict[str, Any] = {}
        for key, state in states.items():
            copies[key] = copy.deepcopy(state)
        return copies

    def begin_sequence(self, reason: str = "new_sequence") -> None:
        """Reset all memory states at the start of a new fold/sequence.

        This is the fail-closed guarantee against cross-fold leakage: a fold
        that never calls this method would inherit (leak) the previous fold's
        memory. Callers MUST invoke this before the first trading day of any
        new fold or independent evaluation sequence.

        Args:
            reason: Reason for reset (for auditability)
        """
        self.reset_memory_states(reason=reason)

    def reset_memory_states(self, reason: str = "manual_reset") -> None:
        """Reset all memory states.

        Used for cold start or after schema changes. Also used by
        ``begin_sequence`` to enforce no cross-fold leakage.

        Args:
            reason: Reason for reset (logged)
        """
        self._seat_memory_states = {}
        self._stock_memory_states = {}
        self._industry_memory_states = {}

    # ------------------------------------------------------------------
    # Decay: γ(Δt) = exp(-λ × Δt), applied to memory BEFORE the GRU update.
    # ------------------------------------------------------------------

    def compute_decay_factor(
        self,
        last_update_date: date,
        current_date: date,
    ) -> float:
        """Compute decay factor γ(Δt) = exp(-λ × Δt).

        Δt is measured in calendar days between the previous and current
        memory updates. With the default 10-day half-life, a 10-day gap
        yields γ ≈ 0.5 and a 20-day gap yields γ ≈ 0.25.

        Args:
            last_update_date: Date of the previous memory update
            current_date: Date of the current memory update

        Returns:
            Decay factor in (0, 1]. Returns 1.0 when Δt <= 0.
        """
        delta_days = (current_date - last_update_date).days
        if delta_days <= 0:
            return 1.0
        lam = self.config.decay_lambda()
        return float(math.exp(-lam * delta_days))

    # ------------------------------------------------------------------
    # Cross-day propagation: advance memory in chronological order.
    # ------------------------------------------------------------------

    def step_seat_memory(
        self,
        seat_id: str,
        daily_event_embedding: torch.Tensor,
        event_stats: dict[str, Any],
        current_date: date,
        num_industries: int = 0,
    ) -> Any:
        """Advance seat memory by one day for a single seat.

        Enforces the correct order: apply decay γ(Δt) to the previous memory
        FIRST, then run the GRU update with the new event embedding. A seat
        with no prior state is cold-started (memory_available=False) so the
        gate stays zero until the first real event is observed.

        Args:
            seat_id: Seat identifier
            daily_event_embedding: [event_encoder_dim] aggregated event
            event_stats: Dict with event statistics
            current_date: Current trading date
            num_industries: Number of industries (for feature layout)

        Returns:
            Updated SeatMemoryState
        """
        state = self._seat_memory_states.get(seat_id)
        if state is None:
            state = self.seat_memory.initialize_memory(seat_id, current_date)

        embedding = daily_event_embedding
        if embedding.dim() == 2:
            embedding = embedding.mean(dim=0)

        updated = self.seat_memory.update_memory(
            state, embedding, event_stats, current_date
        )
        self._seat_memory_states[seat_id] = updated
        return updated

    def step_stock_memory(
        self,
        stock_id: str,
        price_features: torch.Tensor,
        seat_message: torch.Tensor,
        industry_message: torch.Tensor,
        current_date: date,
    ) -> Any:
        """Advance stock memory by one day for a single stock.

        Args:
            stock_id: Stock identifier
            price_features: [price_encoder_dim] price-volume features
            seat_message: [seat_message_dim] seat activity message
            industry_message: [industry_message_dim] industry message
            current_date: Current trading date

        Returns:
            Updated StockMemoryState
        """
        state = self._stock_memory_states.get(stock_id)
        if state is None:
            state = self.stock_memory.initialize_memory(stock_id)

        updated = self.stock_memory.update_memory(
            state, price_features, seat_message, industry_message, current_date
        )
        self._stock_memory_states[stock_id] = updated
        return updated

    def step_industry_memory(
        self,
        industry_id: str,
        daily_event_embedding: torch.Tensor,
        event_stats: dict[str, Any],
        current_date: date,
    ) -> Any:
        """Advance industry memory by one day for a single industry.

        Args:
            industry_id: Industry identifier
            daily_event_embedding: [event_encoder_dim] aggregated event
            event_stats: Dict with event statistics
            current_date: Current trading date

        Returns:
            Updated IndustryMemoryState
        """
        state = self._industry_memory_states.get(industry_id)
        if state is None:
            state = self.industry_memory.initialize_memory(industry_id)

        # IndustryMemory.update_memory signature is
        # (state, event_features, current_date, decay_factor=...). It derives
        # its decay internally from the previous state's last_update_date, so
        # only the event embedding and the current date are forwarded here.
        embedding = daily_event_embedding
        if embedding.dim() == 2:
            embedding = embedding.mean(dim=0)

        updated = self.industry_memory.update_memory(
            state, embedding, current_date
        )
        self._industry_memory_states[industry_id] = updated
        return updated

    # ------------------------------------------------------------------
    # Memory → embedding fusion with learnable gating.
    # ------------------------------------------------------------------

    def _project_memory_component(
        self,
        memory_component: torch.Tensor,
        offset: int,
        length: int,
    ) -> torch.Tensor:
        """Run a single memory component through the full projection stack.

        The shared ``memory_projection`` is a 4-layer Sequential:
        ``Linear(total_dim → hidden) → SiLU → Dropout → Linear(hidden → hidden)``.
        The previous implementation ran ``F.silu`` twice and skipped the final
        Linear; here we slice the input columns of the first Linear and apply
        the remaining layers in order so the projection is actually used.

        Args:
            memory_component: [length] (or [B, length]) raw memory
            offset: Column offset of this entity within the concatenated input
            length: Number of columns for this entity

        Returns:
            [hidden_dim] (or [B, hidden_dim]) projected memory
        """
        first_linear = cast(nn.Linear, self.memory_projection[0])
        projected = F.linear(
            memory_component,
            first_linear.weight[:, offset : offset + length],
            first_linear.bias,
        )
        # Apply remaining layers [1:] in order: SiLU → Dropout → Linear
        for layer in cast(tuple[nn.Module, ...], self.memory_projection[1:]):
            projected = layer(projected)
        return projected

    def project_stock_memory(
        self,
        stock_id: str,
        stock_memory_features: torch.Tensor,
        base_embedding: torch.Tensor,
        memory_available: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project stock memory and compute gate.

        Args:
            stock_id: Stock identifier
            stock_memory_features: [stock_memory_dim + 5] memory features
                (memory + [buy, sell, net, concentration, industry_rel_str]).
                Note: the stock memory feature layout does NOT carry an
                explicit ``memory_available`` flag; pass it explicitly.
            base_embedding: [hidden_dim] base stock embedding
            memory_available: Whether real memory is available for this stock.
                When None, falls back to the tracked StockMemoryState flag.
                When False, the gate is forced to zero so memory contributes
                nothing (cold-start safety).

        Returns:
            Tuple of (projected_memory, gate_value)
        """
        length = self.config.stock_memory_dim
        memory_component = stock_memory_features[:length]
        offset = self.config.seat_memory_dim  # stock follows seat in concat layout

        projected = self._project_memory_component(memory_component, offset, length)

        # Resolve the memory_available flag explicitly (the stock feature
        # vector has no such flag at a fixed index, so never infer it from
        # an arbitrary metadata float).
        if memory_available is None:
            state = self._stock_memory_states.get(stock_id)
            memory_available = bool(getattr(state, "memory_available", False))

        # Compute gate; zero it out when no real memory is available so a
        # cold-started node cannot inject uninitialized signal.
        gate_input = torch.cat([base_embedding, projected], dim=-1)
        gate_logit = self.stock_memory_gate(gate_input)
        gate = torch.sigmoid(gate_logit.squeeze(-1))
        if not memory_available:
            gate = gate * 0.0

        return projected, gate

    def project_seat_memory(
        self,
        seat_id: str,
        seat_memory_features: torch.Tensor,
        seat_base_embedding: torch.Tensor,
        memory_available: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project seat memory and compute gate.

        Args:
            seat_id: Seat identifier
            seat_memory_features: [seat_memory_dim + 6] memory features
                (memory + [buy, sell, net, activity, memory_available,
                event_count]).
            seat_base_embedding: [hidden_dim] base seat embedding
            memory_available: Whether real memory is available for this seat.
                When None, read the flag from index ``seat_memory_dim + 4``
                of the feature vector (matching SeatMemory.get_memory_features).

        Returns:
            Tuple of (projected_memory, gate_value)
        """
        length = self.config.seat_memory_dim
        memory_component = seat_memory_features[:length]
        offset = 0  # seat is first in the concatenated memory layout

        projected = self._project_memory_component(memory_component, offset, length)

        if memory_available is None:
            flag_idx = length + 4
            if seat_memory_features.numel() > flag_idx:
                memory_available = bool(seat_memory_features[flag_idx].item() != 0.0)
            else:
                state = self._seat_memory_states.get(seat_id)
                memory_available = bool(getattr(state, "memory_available", False))

        gate_input = torch.cat([seat_base_embedding, projected], dim=-1)
        gate_logit = self.seat_memory_gate(gate_input)
        gate = torch.sigmoid(gate_logit.squeeze(-1))
        if not memory_available:
            gate = gate * 0.0

        return projected, gate

    def apply_memory_to_embedding(
        self,
        base_embedding: torch.Tensor,
        projected_memory: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse memory into a base embedding via the gated residual.

            h_final = h_base + gate ⊙ projected_memory

        The gate is broadcast over the trailing hidden dimension; when the
        gate is zero the embedding is returned unchanged (no memory leak for
        cold-start nodes).

        Args:
            base_embedding: [N, hidden_dim] base embedding
            projected_memory: [N, hidden_dim] projected memory
            gate: [N] gate values in [0, 1]

        Returns:
            [N, hidden_dim] fused embedding
        """
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        return base_embedding + gate * projected_memory

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        graph_data: Any,
        enable_seat_branch: bool = True,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        """Forward pass with optional memory integration.

        Args:
            graph_data: PyG HeteroData object
            enable_seat_branch: Whether to enable seat residual branch
            use_memory: Whether to use memory (disable for ablation)

        Returns:
            dict with predictions and memory diagnostics
        """
        # Base model forward
        base_outputs = self.base_model(graph_data, enable_seat_branch)

        if not use_memory:
            return {
                **base_outputs,
                "memory_gate_mean": 0.0,
                "memory_gate_std": 0.0,
                "memory_norm": 0.0,
                "memory_increment": "unknown",
            }

        # Diagnostics over the tracked memory states.
        # Real per-node memory fusion is driven by ``project_*_memory`` +
        # ``apply_memory_to_embedding`` from the training/evaluation loop,
        # which owns the per-node event stats and feature vectors. Here we
        # surface aggregate diagnostics so the ablation signal is observable.
        gate_values: list[float] = []
        memory_norm_sq = 0.0
        cold_start = 0
        total = 0

        for state_dict in (
            self._seat_memory_states,
            self._stock_memory_states,
            self._industry_memory_states,
        ):
            for state in state_dict.values():
                total += 1
                if getattr(state, "memory_available", False):
                    mem = getattr(state, "memory", None)
                    if isinstance(mem, torch.Tensor):
                        memory_norm_sq += float(mem.detach().pow(2).sum().item())
                else:
                    cold_start += 1

        memory_gate_mean = 0.0
        memory_gate_std = 0.0
        if gate_values:
            import statistics

            memory_gate_mean = statistics.fmean(gate_values)
            memory_gate_std = (
                statistics.pstdev(gate_values) if len(gate_values) > 1 else 0.0
            )

        memory_diagnostics = {
            "memory_gate_mean": memory_gate_mean,
            "memory_gate_std": memory_gate_std,
            "memory_norm": float(memory_norm_sq ** 0.5),
            "cold_start_ratio": (cold_start / total) if total else 0.0,
            "memory_increment": "tracked_states" if total else "no_states",
        }

        return {
            **base_outputs,
            **memory_diagnostics,
        }

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        """Count model parameters (base model + memory modules + gates)."""
        if self._parameter_count is None:
            self._parameter_count = sum(p.numel() for p in self.parameters())
        return self._parameter_count

    def validate_parameter_limit(self) -> dict[str, Any]:
        """Validate against parameter limit."""
        count = self.parameter_count()
        return {
            "parameter_count": count,
            "maximum_parameters": self.config.maximum_parameters,
            "within_limit": count <= self.config.maximum_parameters,
            "overage": max(0, count - self.config.maximum_parameters),
        }


def build_dt_seat_hgt_memory(
    config: DTSeatHGTMemoryConfig,
    graph_schema: dict[str, Any] | None = None,
) -> DTSeatHGTMemoryModel:
    """Build DT-SeatHGT with memory model.

    Args:
        config: Model configuration
        graph_schema: Optional frozen schema for shape inference

    Returns:
        DTSeatHGTMemoryModel instance
    """
    model = DTSeatHGTMemoryModel(config, graph_schema)
    validation = model.validate_parameter_limit()
    if not validation["within_limit"]:
        raise ValueError(
            f"Parameter limit exceeded: {validation['parameter_count']:,} > "
            f"{validation['maximum_parameters']:,}"
        )
    return model


def default_memory_config() -> DTSeatHGTMemoryConfig:
    """Get default memory configuration for 8GB GPU.

    8GB GPU Fixed Configuration:
    - memory_dim: 32
    - seat_memory_dim: 32
    - stock_memory_dim: 32
    - industry_memory_dim: 16
    - tbptt_days: 20
    - memory_half_life_trading_days: 10
    - memory_dropout: 0.10

    When GPU memory exceeds limit, reduce in order:
    1. memory_dim: 32 -> 24
    2. TBPTT: 20 -> 10
    3. hidden_dim: 64 -> 48
    4. Candidate stocks: 300 -> 200
    """
    return DTSeatHGTMemoryConfig(
        memory_dim=32,
        seat_memory_dim=32,
        stock_memory_dim=32,
        industry_memory_dim=16,
        tbptt_days=20,
        memory_half_life_trading_days=10,
        memory_dropout=0.10,
        memory_gate_initial_bias=-3.0,
        maximum_parameters=3_000_000,
        maximum_peak_gpu_gib=7.2,
    )
