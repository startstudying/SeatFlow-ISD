"""Heterogeneous graph schema definition.

Node types: Stock, Seat, Event, Theme
Edge types: seat_buy_event, seat_sell_event, event_impacts_stock,
stock_belongs_theme, same_stock_repeat, same_seat_relay,
same_coalition_relay, same_theme_diffusion, capital_rotation,
spillover_candidate

Constraints:
- Maximum 2 hops for propagation
- Top-K truncation per relation type
- No global super-node for Theme (Theme is a condition, not a hub)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    STOCK = "Stock"
    SEAT = "Seat"
    EVENT = "Event"
    THEME = "Theme"


class EdgeType(str, Enum):
    SEAT_BUY_EVENT = "seat_buy_event"
    SEAT_SELL_EVENT = "seat_sell_event"
    EVENT_IMPACTS_STOCK = "event_impacts_stock"
    STOCK_BELONGS_THEME = "stock_belongs_theme"
    SAME_STOCK_REPEAT = "same_stock_repeat"
    SAME_SEAT_RELAY = "same_seat_relay"
    SAME_COALITION_RELAY = "same_coalition_relay"
    SAME_THEME_DIFFUSION = "same_theme_diffusion"
    CAPITAL_ROTATION = "capital_rotation"
    SPILLOVER_CANDIDATE = "spillover_candidate"


@dataclass(frozen=True)
class NodeSchema:
    """Schema for a graph node type."""

    node_type: NodeType
    properties: tuple[str, ...]
    required: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeSchema:
    """Schema for a graph edge type."""

    edge_type: EdgeType
    source_type: NodeType
    target_type: NodeType
    properties: tuple[str, ...]
    max_hops: int = 2
    top_k: int = 20
    directed: bool = True


TENSOR_FEATURE_SCHEMA_VERSION = "graph-continuous-v1"


@dataclass(frozen=True)
class TensorFeatureSchema:
    """Versioned whitelist and ordering for continuous tensor features."""

    version: str
    feature_names: tuple[str, ...]


# ── Frozen schema definitions ──────────────────────────────────────────

NODE_SCHEMAS: dict[NodeType, NodeSchema] = {
    NodeType.STOCK: NodeSchema(
        node_type=NodeType.STOCK,
        properties=("stock_id", "ts_code", "exchange", "board"),
        required=("stock_id",),
    ),
    NodeType.SEAT: NodeSchema(
        node_type=NodeType.SEAT,
        properties=("seat_id", "canonical_name", "seat_type"),
        required=("seat_id",),
    ),
    NodeType.EVENT: NodeSchema(
        node_type=NodeType.EVENT,
        properties=("event_id", "trade_date", "reason_codes", "net_amount"),
        required=("event_id",),
    ),
    NodeType.THEME: NodeSchema(
        node_type=NodeType.THEME,
        properties=("theme_id", "theme_name", "theme_type"),
        required=("theme_id",),
    ),
}

EDGE_SCHEMAS: dict[EdgeType, EdgeSchema] = {
    EdgeType.SEAT_BUY_EVENT: EdgeSchema(
        edge_type=EdgeType.SEAT_BUY_EVENT,
        source_type=NodeType.SEAT,
        target_type=NodeType.EVENT,
        properties=("buy_amount", "amount_share", "rank"),
    ),
    EdgeType.SEAT_SELL_EVENT: EdgeSchema(
        edge_type=EdgeType.SEAT_SELL_EVENT,
        source_type=NodeType.SEAT,
        target_type=NodeType.EVENT,
        properties=("sell_amount", "amount_share", "rank"),
    ),
    EdgeType.EVENT_IMPACTS_STOCK: EdgeSchema(
        edge_type=EdgeType.EVENT_IMPACTS_STOCK,
        source_type=NodeType.EVENT,
        target_type=NodeType.STOCK,
        properties=("total_buy", "total_sell", "net_amount"),
    ),
    EdgeType.STOCK_BELONGS_THEME: EdgeSchema(
        edge_type=EdgeType.STOCK_BELONGS_THEME,
        source_type=NodeType.STOCK,
        target_type=NodeType.THEME,
        properties=("membership_type",),
        max_hops=1,
    ),
    EdgeType.SAME_STOCK_REPEAT: EdgeSchema(
        edge_type=EdgeType.SAME_STOCK_REPEAT,
        source_type=NodeType.EVENT,
        target_type=NodeType.EVENT,
        properties=("days_gap",),
    ),
    EdgeType.SAME_SEAT_RELAY: EdgeSchema(
        edge_type=EdgeType.SAME_SEAT_RELAY,
        source_type=NodeType.STOCK,
        target_type=NodeType.STOCK,
        properties=("seat_id", "candidate_score"),
        max_hops=2,
        top_k=20,
    ),
    EdgeType.SAME_COALITION_RELAY: EdgeSchema(
        edge_type=EdgeType.SAME_COALITION_RELAY,
        source_type=NodeType.STOCK,
        target_type=NodeType.STOCK,
        properties=("coalition_score",),
        max_hops=2,
        top_k=20,
    ),
    EdgeType.SAME_THEME_DIFFUSION: EdgeSchema(
        edge_type=EdgeType.SAME_THEME_DIFFUSION,
        source_type=NodeType.STOCK,
        target_type=NodeType.STOCK,
        properties=("theme_id", "diffusion_score"),
        max_hops=2,
        top_k=20,
    ),
    EdgeType.CAPITAL_ROTATION: EdgeSchema(
        edge_type=EdgeType.CAPITAL_ROTATION,
        source_type=NodeType.STOCK,
        target_type=NodeType.STOCK,
        properties=("rotation_score",),
        max_hops=2,
        top_k=20,
    ),
    EdgeType.SPILLOVER_CANDIDATE: EdgeSchema(
        edge_type=EdgeType.SPILLOVER_CANDIDATE,
        source_type=NodeType.STOCK,
        target_type=NodeType.STOCK,
        properties=("candidate_score", "relation_type"),
        max_hops=2,
        top_k=20,
    ),
}


def _tensor_schema(*feature_names: str) -> TensorFeatureSchema:
    return TensorFeatureSchema(
        version=TENSOR_FEATURE_SCHEMA_VERSION,
        feature_names=feature_names,
    )


NODE_TENSOR_FEATURE_SCHEMAS: dict[NodeType, TensorFeatureSchema] = {
    NodeType.STOCK: _tensor_schema("stock_volatility_5d"),
    NodeType.SEAT: _tensor_schema("seat_win_rate"),
    NodeType.EVENT: _tensor_schema("event_net_amount"),
    NodeType.THEME: _tensor_schema(),
}

EDGE_TENSOR_FEATURE_SCHEMAS: dict[EdgeType, TensorFeatureSchema] = {
    EdgeType.SEAT_BUY_EVENT: _tensor_schema("buy_amount"),
    EdgeType.SEAT_SELL_EVENT: _tensor_schema("sell_amount"),
    EdgeType.EVENT_IMPACTS_STOCK: _tensor_schema("net_amount"),
    EdgeType.STOCK_BELONGS_THEME: _tensor_schema(),
    EdgeType.SAME_STOCK_REPEAT: _tensor_schema("days_gap"),
    EdgeType.SAME_SEAT_RELAY: _tensor_schema("candidate_score"),
    EdgeType.SAME_COALITION_RELAY: _tensor_schema("coalition_score"),
    EdgeType.SAME_THEME_DIFFUSION: _tensor_schema("diffusion_score"),
    EdgeType.CAPITAL_ROTATION: _tensor_schema("rotation_score"),
    EdgeType.SPILLOVER_CANDIDATE: _tensor_schema("candidate_score"),
}


def get_node_schema(node_type: NodeType) -> NodeSchema:
    return NODE_SCHEMAS[node_type]


def get_edge_schema(edge_type: EdgeType) -> EdgeSchema:
    return EDGE_SCHEMAS[edge_type]


def get_node_tensor_feature_schema(node_type: NodeType) -> TensorFeatureSchema:
    return NODE_TENSOR_FEATURE_SCHEMAS[node_type]


def get_edge_tensor_feature_schema(edge_type: EdgeType) -> TensorFeatureSchema:
    return EDGE_TENSOR_FEATURE_SCHEMAS[edge_type]


def max_propagation_hops() -> int:
    """Global maximum hops for propagation edges."""
    return 2


def default_top_k() -> int:
    """Default Top-K truncation for propagation edges."""
    return 20
