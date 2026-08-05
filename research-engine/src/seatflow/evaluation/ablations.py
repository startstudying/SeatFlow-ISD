"""Ablation matrix definitions (Task 18).

Required variants:
- Base
- Amount Only
- Seat Type
- Anonymous Seat
- Real Seat Direct
- No Graph
- Shuffled Graph
- Real Graph Spillover
- Full FDG
- Full FDG without Relation State routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AblationVariant(str, Enum):
    """All required ablation variants."""

    BASE = "base"
    AMOUNT_ONLY = "amount_only"
    SEAT_TYPE = "seat_type"
    ANONYMOUS_SEAT = "anonymous_seat"
    REAL_SEAT_DIRECT = "real_seat_direct"
    NO_GRAPH = "no_graph"
    SHUFFLED_GRAPH = "shuffled_graph"
    REAL_GRAPH_SPILLOVER = "real_graph_spillover"
    FULL_FDG = "full_fdg"
    FULL_FDG_NO_RS = "full_fdg_no_rs"  # without Relation State routing


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for a single ablation variant."""

    variant: AblationVariant
    use_base: bool = True
    use_direct: bool = False
    use_spillover: bool = False
    use_relation_state: bool = True
    use_seat_type: bool = True
    use_seat_amount: bool = True
    use_anonymous_seat: bool = False
    use_shuffled_graph: bool = False
    use_real_graph: bool = True
    description: str = ""


# Pre-defined configurations for each variant
ABLATION_CONFIGS: dict[AblationVariant, AblationConfig] = {
    AblationVariant.BASE: AblationConfig(
        variant=AblationVariant.BASE,
        use_base=True,
        description="Market features only, no seat/event/graph info",
    ),
    AblationVariant.AMOUNT_ONLY: AblationConfig(
        variant=AblationVariant.AMOUNT_ONLY,
        use_base=True,
        use_direct=True,
        use_seat_amount=True,
        use_seat_type=False,
        description="Direct effect using only seat buy/sell amounts",
    ),
    AblationVariant.SEAT_TYPE: AblationConfig(
        variant=AblationVariant.SEAT_TYPE,
        use_base=True,
        use_direct=True,
        use_seat_type=True,
        use_seat_amount=False,
        description="Direct effect using only seat type (broker/institution/northbound)",
    ),
    AblationVariant.ANONYMOUS_SEAT: AblationConfig(
        variant=AblationVariant.ANONYMOUS_SEAT,
        use_base=True,
        use_direct=True,
        use_anonymous_seat=True,
        description="Direct effect with seat identity removed (anonymous)",
    ),
    AblationVariant.REAL_SEAT_DIRECT: AblationConfig(
        variant=AblationVariant.REAL_SEAT_DIRECT,
        use_base=True,
        use_direct=True,
        description="Full Direct effect with real seat identity",
    ),
    AblationVariant.NO_GRAPH: AblationConfig(
        variant=AblationVariant.NO_GRAPH,
        use_base=True,
        use_direct=True,
        use_spillover=False,
        use_real_graph=False,
        description="No graph propagation at all",
    ),
    AblationVariant.SHUFFLED_GRAPH: AblationConfig(
        variant=AblationVariant.SHUFFLED_GRAPH,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_shuffled_graph=True,
        description="Spillover with shuffled graph edges (placebo)",
    ),
    AblationVariant.REAL_GRAPH_SPILLOVER: AblationConfig(
        variant=AblationVariant.REAL_GRAPH_SPILLOVER,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_real_graph=True,
        description="Spillover with real graph propagation",
    ),
    AblationVariant.FULL_FDG: AblationConfig(
        variant=AblationVariant.FULL_FDG,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_relation_state=True,
        description="Full SeatFlow-FDG with Relation State routing",
    ),
    AblationVariant.FULL_FDG_NO_RS: AblationConfig(
        variant=AblationVariant.FULL_FDG_NO_RS,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_relation_state=False,
        description="Full FDG without Relation State routing (uniform message passing)",
    ),
}


def get_ablation_config(variant: AblationVariant) -> AblationConfig:
    """Get the configuration for an ablation variant."""
    return ABLATION_CONFIGS[variant]


def all_variants() -> list[AblationVariant]:
    """Return all ablation variants in order."""
    return list(AblationVariant)
