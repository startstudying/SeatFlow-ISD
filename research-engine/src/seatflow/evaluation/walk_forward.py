"""Compatibility exports for the canonical dataset walk-forward splitter."""

from __future__ import annotations

from seatflow.dataset.splits import (
    CycleDisposition,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardPlan,
    create_walk_forward_plan,
    create_walk_forward_splits,
    validate_walk_forward_plan,
    verify_cycle_atomicity,
    verify_embargo,
)

__all__ = [
    "CycleDisposition",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardPlan",
    "create_walk_forward_plan",
    "create_walk_forward_splits",
    "validate_walk_forward_plan",
    "verify_cycle_atomicity",
    "verify_embargo",
]
