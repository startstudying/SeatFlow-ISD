"""DT-SeatHGT v3 model for R5.

This module implements the core R5 model architecture:
- Stock Temporal Encoder (GRU)
- Stock Static Encoder
- Base Relation Branch (Industry→Stock, Stock→Stock)
- Seat Residual Branch (Seat→Stock buys/sells only)
- Relation Router (seat_gate)
- Multi-task heads

Architecture:
    Stock 20-day sequence
            │
            ▼
    StockTemporalEncoder (GRU)
            │
            ▼
    StockStaticEncoder (MLP)
            │
            ▼
    BaseFusion → base_embedding
            │
    ┌─────┴─────┐
    ▼             ▼
Industry/Stock  Seat-Stock
Relations      Residual
    │             │
    └─────┬─────┘
            ▼
    RelationRouter
            │
            ▼
    Final Embedding
            │
    ┌─────┼─────┐
    ▼     ▼     ▼
   Rank Return Direction
   Head Head  Head
            │
            ▼
    Uncertainty
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DTSeatHGTv3Config:
    """Configuration for DT-SeatHGT v3 model."""

    # Stock branch
    hidden_dim: int = 64
    stock_gru_hidden_dim: int = 48
    stock_gru_layers: int = 1
    stock_static_hidden_dim: int = 32
    stock_sequence_length: int = 20
    stock_sequence_feature_dim: int = 5  # 5 reliable temporal features
    stock_static_feature_count: int = 8  # 8 stock static features

    # Seat dynamic soft profile
    seat_style_count: int = 5
    seat_style_embedding_dim: int = 32
    # Whether the SeatStyleEncoder output is wired into the seat node
    # representation. False (default) keeps E3 behavior (style encoder is
    # constructed but its output is not injected); True enables E4 behavior
    # where style_embedding is added to x_dict["seat"].
    use_seat_style_encoder: bool = False
    seat_feature_dim: int = 32  # Seat node feature dimension from graph
    # Industry node feature dimension. None => fall back to hidden_dim (the
    # historical default). Set explicitly when there is no frozen graph schema
    # to infer from (e.g. test fixtures) so industry_projection is built with
    # the correct input dim instead of guessing hidden_dim (T2 leftover #2).
    industry_feature_dim: int | None = None

    # Relation branches
    num_heads: int = 2
    base_relation_layers: int = 1
    seat_relation_layers: int = 1

    # Dropout
    dropout: float = 0.15
    relation_dropout: float = 0.10

    # Seat gate
    seat_gate_initial_bias: float = -2.0

    # Experiment-level relation switches (Phase 2). When False the
    # corresponding base-relation edges are skipped in forward, so E0
    # (stock-only baseline) differs from E1 (industry+stock relations).
    use_industry_relations: bool = True
    use_stock_relations: bool = True

    # Target
    target_node_type: str = "stock"
    direction_classes: int = 2  # 2=binary (BCE), >2=multi-class (CE)

    # Maximum parameters
    maximum_parameters: int = 3_000_000

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim > 128:
            raise ValueError("hidden_dim must be in [1, 128]")
        if self.stock_gru_hidden_dim <= 0:
            raise ValueError("stock_gru_hidden_dim must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("num_heads must divide hidden_dim evenly")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.seat_gate_initial_bias > 0:
            raise ValueError("seat_gate_initial_bias must be <= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "stock_gru_hidden_dim": self.stock_gru_hidden_dim,
            "stock_gru_layers": self.stock_gru_layers,
            "stock_static_hidden_dim": self.stock_static_hidden_dim,
            "stock_sequence_length": self.stock_sequence_length,
            "stock_sequence_feature_dim": self.stock_sequence_feature_dim,
            "stock_static_feature_count": self.stock_static_feature_count,
            "seat_style_count": self.seat_style_count,
            "seat_style_embedding_dim": self.seat_style_embedding_dim,
            "seat_feature_dim": self.seat_feature_dim,
            "industry_feature_dim": self.industry_feature_dim,
            "num_heads": self.num_heads,
            "base_relation_layers": self.base_relation_layers,
            "seat_relation_layers": self.seat_relation_layers,
            "dropout": self.dropout,
            "relation_dropout": self.relation_dropout,
            "seat_gate_initial_bias": self.seat_gate_initial_bias,
            "target_node_type": self.target_node_type,
            "direction_classes": self.direction_classes,
            "maximum_parameters": self.maximum_parameters,
        }

    def fingerprint(self) -> str:
        """Generate hash of config for checkpoint binding."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()


class DTSeatHGTv3Model(nn.Module):
    """DT-SeatHGT v3 model with seat residual architecture.

    Single constructor - all layers created at initialization time.
    Real heterogeneous edge propagation via typed relation branches.
    """

    def __init__(
        self,
        config: DTSeatHGTv3Config,
        graph_schema: dict[str, Any] | None = None,
    ) -> None:
        """Initialize model.

        Args:
            config: Model configuration
            graph_schema: Frozen schema with node_types, edge_types, feature_dims.
                Can be a dict or a TensorizedGraph with metadata containing frozen_graph_schema.
        """
        super().__init__()
        self.config = config
        self._parameter_count: int | None = None

        # Infer feature dimensions from graph schema if provided
        # Handle both dict and TensorizedGraph (extract from metadata)
        seat_feature_dim = config.seat_feature_dim
        # Explicit config field wins when no schema can be inferred (T2 #2).
        industry_feature_dim = (
            config.industry_feature_dim
            if config.industry_feature_dim is not None
            else config.hidden_dim
        )
        # Seat edge attribute dims inferred from frozen schema (V4 gap fix).
        # None => SeatRelationBranch falls back to its hardcoded default (7).
        seat_edge_attr_dims: dict[str, int] | None = None
        if graph_schema is not None:
            schema_dict: dict[str, Any] = {}
            if isinstance(graph_schema, dict):
                # Direct dict
                schema_dict = graph_schema
            elif hasattr(graph_schema, "_metadata"):
                # Our mock data format - _metadata attribute
                schema_dict = graph_schema._metadata.get("frozen_graph_schema", {})
                # Also check feature_schema which contains actual feature names
                if not schema_dict and "_metadata" in dir(graph_schema):
                    metadata = getattr(graph_schema, "_metadata", {})
                    if "feature_schema" in metadata:
                        schema_dict = {"node_feature_schema": metadata.get("feature_schema", {})}
            elif hasattr(graph_schema, "metadata"):
                # PyG HeteroData - metadata property
                metadata = getattr(graph_schema, "metadata", {})
                if isinstance(metadata, dict):
                    schema_dict = metadata.get("frozen_graph_schema", {})
                    # Also check feature_schema which contains actual feature names
                    if not schema_dict and "feature_schema" in metadata:
                        schema_dict = {"node_feature_schema": metadata.get("feature_schema", {})}

            if schema_dict:
                # Try node_types.seat.feature_dim first
                node_types = schema_dict.get("node_types", {})
                if isinstance(node_types, dict):
                    seat_feature_dim = node_types.get("seat", {}).get(
                        "feature_dim", seat_feature_dim
                    )
                    industry_feature_dim = node_types.get("industry", {}).get(
                        "feature_dim", industry_feature_dim
                    )

                # Infer seat feature dim from node_feature_schema if available
                seat_schema = schema_dict.get("node_feature_schema", {}).get("seat", ())
                if seat_schema and isinstance(seat_schema, (list, tuple)):
                    seat_feature_dim = len(seat_schema)

                # Industry feature dim
                industry_schema = schema_dict.get("node_feature_schema", {}).get("industry", ())
                if industry_schema and isinstance(industry_schema, (list, tuple)):
                    industry_feature_dim = len(industry_schema)

                # Seat edge attribute dims from frozen schema. The schema stores
                # edge attrs keyed by "src::rel::dst"; map PyG relation names to
                # the model's internal seat relation names. Falls back to 7
                # (SeatRelationBranch default) when schema has no seat edges.
                raw_edge_attr_schema = schema_dict.get("edge_attr_schema", {})
                if isinstance(raw_edge_attr_schema, dict) and raw_edge_attr_schema:
                    pyg_to_seat_rel = {"buys": "seat_buy_stock", "sells": "seat_sell_stock"}
                    inferred: dict[str, int] = {}
                    for edge_key, attr_schema in raw_edge_attr_schema.items():
                        if not isinstance(edge_key, str) or "::" not in edge_key:
                            continue
                        src_type, rel_name, _dst_type = edge_key.split("::", 2)
                        if src_type != "seat":
                            continue
                        seat_rel = pyg_to_seat_rel.get(rel_name, rel_name)
                        if seat_rel not in ("seat_buy_stock", "seat_sell_stock"):
                            continue
                        if isinstance(attr_schema, (list, tuple)) and attr_schema:
                            inferred[seat_rel] = len(attr_schema)
                    if inferred:
                        # Both seat relations must be present for the encoder to
                        # be built consistently; fill any missing one from the
                        # other (they share the same attr schema in practice).
                        for rel in ("seat_buy_stock", "seat_sell_stock"):
                            if rel not in inferred and inferred:
                                inferred[rel] = next(iter(inferred.values()))
                        seat_edge_attr_dims = inferred

        # Live-graph shape inference: when a PyG HeteroData graph is passed
        # directly (as _evaluate_checkpoint_on_test does via
        # build_dt_seat_hgt_v3(config, first_graph)), read the real stock and
        # seat node feature dims from the tensors so the encoders match the
        # data regardless of what the config defaults say. This is what makes
        # the real-data runner work with V2 snapshots (16-dim stock.x,
        # 18-dim seat.x) instead of the config defaults (8 / 32).
        live_graph = None
        tensorized_metadata: dict[str, Any] | None = None
        if graph_schema is not None and hasattr(graph_schema, "node_types"):
            live_graph = graph_schema
        elif (
            graph_schema is not None
            and hasattr(graph_schema, "metadata")
            and isinstance(getattr(graph_schema, "metadata", None), dict)
        ):
            # TensorizedGraph: PyG HeteroData lives in .data; shape info lives
            # in .metadata (frozen_graph_schema, node_shapes, edge_attr_schema).
            # Use the underlying HeteroData for tensor reads when available.
            live_graph = getattr(graph_schema, "data", None)
            tensorized_metadata = graph_schema.metadata
        if live_graph is not None and hasattr(live_graph, "node_types"):
            try:
                stock_x_live = live_graph["stock"].x
                if stock_x_live is not None and stock_x_live.numel() > 0:
                    stock_static_feature_count_live = int(stock_x_live.size(-1))
                else:
                    stock_static_feature_count_live = None
            except (KeyError, AttributeError, IndexError):
                stock_static_feature_count_live = None
            try:
                seat_x_live = live_graph["seat"].x
                if seat_x_live is not None and seat_x_live.numel() > 0:
                    seat_feature_dim = int(seat_x_live.size(-1))
            except (KeyError, AttributeError, IndexError):
                pass
            # Industry feature dim: also infer from live graph (T2 #2) when
            # frozen schema is unavailable. Real graphs may have empty industry
            # (0 nodes, 1-dim placeholder), which the schema-inference block above
            # handles; this live-graph path catches the "schema=None, graph has
            # industry" case.
            try:
                industry_x_live = live_graph["industry"].x
                if industry_x_live is not None and industry_x_live.numel() > 0:
                    # Only use live inference if config didn't explicitly set it
                    # (explicit config wins). The schema-inference block above
                    # already handles the "schema present" case.
                    if config.industry_feature_dim is None:
                        industry_feature_dim = int(industry_x_live.size(-1))
            except (KeyError, AttributeError, IndexError):
                pass
            # Seat edge attribute dims: read directly from the live graph's
            # seat→stock edge_attr tensors. Real V2 seat buys/sells edge_attr
            # is 14-dim, not the SeatRelationBranch default of 7.
            if seat_edge_attr_dims is None:
                live_seat_edge_dims: dict[str, int] = {}
                pyg_to_seat_rel = {"buys": "seat_buy_stock", "sells": "seat_sell_stock"}
                for edge_key in getattr(live_graph, "edge_types", []):
                    src_type, rel_name, _dst_type = edge_key
                    if src_type != "seat" or rel_name not in pyg_to_seat_rel:
                        continue
                    try:
                        ea = live_graph[edge_key].edge_attr
                        if ea is not None and ea.numel() > 0 and ea.dim() >= 2:
                            live_seat_edge_dims[pyg_to_seat_rel[rel_name]] = int(ea.size(-1))
                    except (KeyError, AttributeError, IndexError):
                        pass
                if live_seat_edge_dims:
                    # Fill any missing seat relation from the other (they share
                    # the same attr schema in practice).
                    for rel in ("seat_buy_stock", "seat_sell_stock"):
                        if rel not in live_seat_edge_dims and live_seat_edge_dims:
                            live_seat_edge_dims[rel] = next(iter(live_seat_edge_dims.values()))
                    seat_edge_attr_dims = live_seat_edge_dims
        else:
            stock_static_feature_count_live = None
        # Metadata fallback for TensorizedGraph: when the underlying HeteroData
        # is unavailable or a node type is empty (e.g. 0 industry nodes), read
        # feature dims from frozen_graph_schema.node_feature_schema / node_shapes
        # so encoders still match the real V2 feature width.
        if tensorized_metadata is not None and stock_static_feature_count_live is None:
            fgs_meta = tensorized_metadata.get("frozen_graph_schema", {})
            if isinstance(fgs_meta, dict):
                stock_schema = fgs_meta.get("node_feature_schema", {}).get("stock", ())
                if isinstance(stock_schema, (list, tuple)) and stock_schema:
                    stock_static_feature_count_live = len(stock_schema)
            if stock_static_feature_count_live is None:
                node_shapes_meta = tensorized_metadata.get("node_shapes", {})
                stock_shape = (
                    node_shapes_meta.get("stock")
                    if isinstance(node_shapes_meta, dict)
                    else None
                )
                if isinstance(stock_shape, (list, tuple)) and len(stock_shape) >= 2:
                    stock_static_feature_count_live = int(stock_shape[1])
        stock_static_feature_count = (
            stock_static_feature_count_live
            if stock_static_feature_count_live is not None
            else config.stock_static_feature_count
        )

        # ═══════════════════════════════════════════════════════════
        # Stock Branch
        # ═══════════════════════════════════════════════════════════

        # Stock Temporal Encoder (GRU)
        self.stock_gru = nn.GRU(
            input_size=config.stock_sequence_feature_dim,
            hidden_size=config.stock_gru_hidden_dim,
            num_layers=config.stock_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Stock Static Encoder (all features in constructor)
        self.stock_static_mlp = nn.Sequential(
            nn.Linear(stock_static_feature_count, config.stock_static_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # Base Fusion: GRU output + Static → hidden_dim
        self.base_projection = nn.Linear(
            config.stock_gru_hidden_dim + config.stock_static_hidden_dim,
            config.hidden_dim,
        )

        # ═══════════════════════════════════════════════════════════
        # Relation Branches
        # ═══════════════════════════════════════════════════════════

        # Import here to avoid circular imports
        from seatflow.models.layers.edge_gated_relation import (
            BaseRelationBranch,
            SeatRelationBranch,
        )

        # Base Relation Branch: Industry→Stock, Stock→Stock
        self.base_relation_branch = BaseRelationBranch(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.relation_dropout,
        )

        # Seat Relation Branch: Seat→Stock buys/sells
        self.seat_relation_branch = SeatRelationBranch(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.relation_dropout,
            use_edge_gate=True,
            edge_attr_dims=seat_edge_attr_dims,
        )

        # Seat projection - maps seat features to hidden_dim. Real V2 seat
        # features include raw monetary amounts (seat_net_buy_amount etc.)
        # with magnitudes up to ~1e10, so a LayerNorm after the projection
        # is required for numerical stability — without it the seat attention
        # logits explode and backward produces NaN on the very first step.
        self.seat_projection = nn.Linear(seat_feature_dim, config.hidden_dim)
        self.seat_norm = nn.LayerNorm(config.hidden_dim)

        # Industry projection - maps industry features to hidden_dim
        self.industry_projection = nn.Linear(industry_feature_dim, config.hidden_dim)

        # Seat style encoder (standalone module)
        from seatflow.models.seat_style_encoder import SeatStyleEncoder, SeatStyleEncoderConfig

        seat_encoder_config = SeatStyleEncoderConfig(
            seat_style_count=config.seat_style_count,
            seat_style_embedding_dim=config.seat_style_embedding_dim,
            feature_dim=seat_feature_dim,
            hidden_dim=config.hidden_dim // 2,
        )
        self.seat_style_encoder = SeatStyleEncoder(seat_encoder_config)
        # Projection from style_embedding_dim -> hidden_dim so the style
        # prototype embedding can be added to the seat node representation
        # (x_dict["seat"]) when use_seat_style_encoder is True (E4).
        self.style_to_seat = nn.Linear(config.seat_style_embedding_dim, config.hidden_dim)

        # ═══════════════════════════════════════════════════════════
        # Seat Gate Router
        # ═══════════════════════════════════════════════════════════

        # Gate input: base_embedding + seat_delta + seat_metadata
        gate_input_dim = config.hidden_dim * 2 + 3  # base + seat_delta + metadata
        self.seat_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        # Initialize bias to -2.0 for conservative gate
        self.seat_gate_mlp[-1].bias.data.fill_(  # type: ignore[operator]
            config.seat_gate_initial_bias
        )

        # ═══════════════════════════════════════════════════════════
        # Output Heads
        # ═══════════════════════════════════════════════════════════

        # Base heads (no seat influence)
        self.base_rank_head = nn.Linear(config.hidden_dim, 1)
        self.base_return_head = nn.Linear(config.hidden_dim, 1)

        # Seat delta heads (incremental contribution)
        self.seat_rank_delta_head = nn.Linear(config.hidden_dim, 1)
        self.seat_return_delta_head = nn.Linear(config.hidden_dim, 1)

        # Final heads (on combined embedding)
        self.final_rank_head = nn.Linear(config.hidden_dim, 1)
        self.final_return_head = nn.Linear(config.hidden_dim, 1)

        # Direction head: 1 output for binary (BCE), N outputs for multi-class (CE)
        num_dir = config.direction_classes
        num_direction_outputs = 1 if num_dir == 2 else max(2, num_dir)
        self.direction_head = nn.Linear(config.hidden_dim, num_direction_outputs)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def _extract_node_features(
        self,
        graph_data: Any,
        enable_seat_branch: bool,
    ) -> dict[str, torch.Tensor]:
        """Extract and process node features.

        Args:
            graph_data: PyG HeteroData object
            enable_seat_branch: Whether to extract seat features

        Returns:
            Dict of node_type -> feature tensor
        """
        x_dict: dict[str, torch.Tensor] = {}

        # Stock features
        stock_x = graph_data["stock"].x  # [N_stock, static_dim]

        # Stock temporal sequence
        if hasattr(graph_data["stock"], "sequence"):
            stock_seq = graph_data["stock"].sequence  # [N_stock, 20, F]
            stock_mask = graph_data["stock"].sequence_mask  # [N_stock, 20]

            # GRU encoding
            gru_out, _ = self.stock_gru(stock_seq)
            # Take last valid output
            last_idx = (stock_mask.sum(dim=1) - 1).clamp(min=0).long()
            stock_gru_out = gru_out[torch.arange(len(last_idx)), last_idx]
        else:
            stock_gru_out = torch.zeros(
                stock_x.size(0),
                self.config.stock_gru_hidden_dim,
                device=stock_x.device,
            )

        # Static encoding
        stock_static = self.stock_static_mlp(stock_x)

        # Fuse GRU and static
        base_input = torch.cat([stock_gru_out, stock_static], dim=-1)
        base_embedding = torch.relu(self.base_projection(base_input))
        base_embedding = self.layer_norm(self.dropout(base_embedding))

        x_dict["stock"] = base_embedding

        # Seat features (if enabled and seat nodes exist)
        # Note: PyG HeteroData's __contains__ doesn't work for node types,
        # so we check node_types or num_nodes_dict instead
        seat_exists = (
            "seat" in getattr(graph_data, "node_types", [])
            or "seat" in getattr(graph_data, "num_nodes_dict", {})
        )
        if enable_seat_branch and seat_exists:
            seat_x = graph_data["seat"].x  # [N_seat, seat_feature_dim]
            # Project + normalize seat features to hidden_dim. LayerNorm is
            # critical here: raw seat features contain monetary amounts with
            # magnitudes ~1e10 that would otherwise explode attention logits.
            seat_repr = self.seat_norm(self.seat_projection(seat_x))
            # E4: wire SeatStyleEncoder output into the seat representation.
            # style_embedding ([N_seat, style_embedding_dim]) is projected to
            # hidden_dim and added so seat_delta carries style information.
            # Without this the style encoder is dead code (E4 == E3).
            if self.config.use_seat_style_encoder:
                style_output = self.seat_style_encoder(seat_x)
                style_emb = style_output["style_embedding"]  # [N_seat, D_style]
                seat_repr = seat_repr + self.style_to_seat(style_emb)
            x_dict["seat"] = seat_repr

        # Industry features (if industry nodes exist)
        num_nodes_dict = getattr(graph_data, "num_nodes_dict", {})
        industry_exists = (
            "industry" in getattr(graph_data, "node_types", [])
            and num_nodes_dict.get("industry", 0) > 0
        )
        if industry_exists:
            industry_x = graph_data["industry"].x  # [N_industry, industry_feature_dim]
            # Project industry features to hidden_dim using the registered projection
            x_dict["industry"] = self.industry_projection(industry_x)

        return x_dict

    def _get_edge_index_dict(
        self,
        graph_data: Any,
        allowed_relations: set[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """Extract edge indices for allowed relations.

        Args:
            graph_data: PyG HeteroData object
            allowed_relations: Set of allowed (src, rel, dst) tuples

        Returns:
            Dict of (src, rel, dst) -> edge_index tensor
        """
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        # Build reverse mapping: PyG relation name -> model relation name
        # PyG uses various relation names, model uses consistent internal names
        pyg_to_model_relation = {
            # Seat relations
            "buys": "seat_buy_stock",  # PyG uses "buys", model uses "seat_buy_stock"
            "sells": "seat_sell_stock",
            # Industry relations
            "belongs_to": "industry_to_stock",
            "in": "industry_to_stock",
            "belongs": "industry_to_stock",
            # Stock self-relations
            "contains": "stock_to_stock",
            "similar": "stock_to_stock",
        }

        # Safely check if graph has edges (PyG raises on hasattr without edges)
        try:
            raw_edge_index_dict = graph_data.edge_index_dict
        except KeyError:
            # No edges in graph
            return edge_index_dict

        # PyG stores edges as (src, rel, dst) tuples
        for rel_key, edge_index in raw_edge_index_dict.items():
            src_type, rel_name, dst_type = rel_key

            # Check if this exact tuple is allowed
            if rel_key in allowed_relations:
                edge_index_dict[rel_key] = edge_index
                continue

            # Try mapping the relation name
            mapped_rel_name = pyg_to_model_relation.get(rel_name)
            if mapped_rel_name:
                mapped_key = (src_type, mapped_rel_name, dst_type)
                if mapped_key in allowed_relations:
                    # Use the model's internal key (mapped_key), not the PyG key (rel_key)
                    edge_index_dict[mapped_key] = edge_index

        return edge_index_dict

    def _get_edge_attr_dict(
        self,
        graph_data: Any,
        allowed_relations: set[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], torch.Tensor] | None:
        """Extract edge attributes for allowed relations.

        Args:
            graph_data: PyG HeteroData object
            allowed_relations: Set of allowed (src, rel, dst) tuples

        Returns:
            Dict of (src, rel, dst) -> edge_attr tensor, or None
        """
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        # Try to get edge_attr_dict from PyG
        try:
            raw_edge_attr_dict = graph_data.edge_attr_dict
        except (KeyError, AttributeError):
            raw_edge_attr_dict = {}

        # Build mapping from model relation names to PyG relation names
        model_to_pyg_relation: dict[str, str] = {
            "seat_buy_stock": "buys",
            "seat_sell_stock": "sells",
            "industry_to_stock": "belongs_to",
            "stock_to_stock": "contains",
        }

        for rel_key in allowed_relations:
            src_type, model_rel_name, dst_type = rel_key

            # Check if exact key exists in raw dict
            if rel_key in raw_edge_attr_dict:
                edge_attr_dict[rel_key] = raw_edge_attr_dict[rel_key]
                continue

            # Try to find the PyG key that maps to this model relation
            pyg_rel_name = model_to_pyg_relation.get(model_rel_name)
            if pyg_rel_name:
                # Try various PyG relation names
                for alt_pyg_name in [pyg_rel_name, model_rel_name]:
                    mapped_key = (src_type, alt_pyg_name, dst_type)
                    if mapped_key in raw_edge_attr_dict:
                        edge_attr_dict[rel_key] = raw_edge_attr_dict[mapped_key]
                        break

        return edge_attr_dict if edge_attr_dict else None

    def _compute_seat_metadata(
        self,
        stock_idx: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute seat metadata for gate input.

        Args:
            stock_idx: [batch_size] indices of stocks
            edge_index: [2, num_seat_stock_edges] seat→stock edges

        Returns:
            Tuple of:
            - [batch_size, 3] metadata tensor (seat_in_degree, named_ratio, proxy_ratio)
            - [batch_size] bool tensor indicating which stocks have seat connections
        """
        batch_size = stock_idx.size(0)
        device = stock_idx.device

        metadata = torch.zeros(batch_size, 3, device=device)
        has_seat_connection = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if edge_index is None or edge_index.size(1) == 0:
            # No seat edges
            return metadata, has_seat_connection

        # Count seat edges per stock
        dst_idx = edge_index[1]  # destination (stock) indices
        edge_count = torch.bincount(dst_idx, minlength=batch_size)

        # Mark stocks that have seat connections
        has_seat_connection = edge_count > 0

        # Seat in-degree (normalized by max, avoiding div by zero)
        max_count = edge_count.float().max().clamp(min=1.0)
        metadata[:, 0] = edge_count.float() / max_count

        # Named seat ratio (placeholder - would need seat type info)
        metadata[:, 1] = 0.5

        # Proxy ratio (placeholder - would need seat type info)
        metadata[:, 2] = 0.5

        return metadata, has_seat_connection

    def forward(
        self,
        graph_data: Any,
        enable_seat_branch: bool = True,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            graph_data: PyG HeteroData object
            enable_seat_branch: Whether to enable seat residual branch

        Returns:
            dict with:
            - base_rank_score
            - base_return_prediction
            - seat_rank_delta
            - seat_return_delta
            - seat_gate
            - final_rank_score
            - final_return_prediction
            - direction_logit
            - uncertainty
            - model_valid (always False)
        """
        # ═══════════════════════════════════════════════════════════
        # Extract Node Features
        # ═══════════════════════════════════════════════════════════
        x_dict = self._extract_node_features(graph_data, enable_seat_branch)
        base_embedding = x_dict["stock"]  # [N_stock, hidden_dim]

        # ═══════════════════════════════════════════════════════════
        # Base Relation Branch (Industry→Stock, Stock→Stock)
        # ═══════════════════════════════════════════════════════════
        # Phase 2: filter allowed base relations by experiment config so E0
        # (stock-only) and E1 (industry+stock) produce different graphs.
        allowed_base_relations: set[tuple[str, str, str]] = set()
        if self.config.use_industry_relations:
            allowed_base_relations |= {("industry", "industry_to_stock", "stock")}
        if self.config.use_stock_relations:
            allowed_base_relations |= {("stock", "stock_to_stock", "stock")}
        base_edge_index = self._get_edge_index_dict(
            graph_data,
            allowed_base_relations,
        )
        base_edge_attr = self._get_edge_attr_dict(
            graph_data,
            allowed_base_relations,
        )

        if base_edge_index:
            base_result = self.base_relation_branch(
                x_dict=x_dict,
                edge_index_dict=base_edge_index,
                edge_attr_dict=base_edge_attr,
            )
            if "stock" in base_result:
                base_embedding = base_embedding + base_result["stock"]

        # ═══════════════════════════════════════════════════════════
        # Base Predictions
        # ═══════════════════════════════════════════════════════════
        base_rank = self.base_rank_head(base_embedding).squeeze(-1)
        base_return = self.base_return_head(base_embedding).squeeze(-1)

        # ═══════════════════════════════════════════════════════════
        # Seat Residual Branch (Seat→Stock)
        # ═══════════════════════════════════════════════════════════
        # Initialize seat edge index early so it's available for diagnostics
        seat_edge_index: dict[tuple[str, str, str], torch.Tensor] = {}
        seat_edge_attr: dict[tuple[str, str, str], torch.Tensor] | None = None

        if enable_seat_branch and "seat" in x_dict:
            # Get seat edge information
            seat_edge_index = self._get_edge_index_dict(
                graph_data,
                self.seat_relation_branch.allowed_edge_type_tuples,
            )
            seat_edge_attr = self._get_edge_attr_dict(
                graph_data,
                self.seat_relation_branch.allowed_edge_type_tuples,
            )

            if seat_edge_index:
                # Process seat relations
                seat_result = self.seat_relation_branch(
                    x_dict=x_dict,
                    edge_index_dict=seat_edge_index,
                    edge_attr_dict=seat_edge_attr,
                )
                seat_delta = seat_result.get("stock", torch.zeros_like(base_embedding))

                # Seat style encoding (SeatStyleEncoder handles its own projection)
                seat_features = graph_data["seat"].x  # Raw seat features
                style_output = self.seat_style_encoder(seat_features)
                seat_style_probs = style_output["style_probs"]  # [N_seat, K]
            else:
                # No seat edges - zero delta
                seat_delta = torch.zeros_like(base_embedding)
                seat_style_probs = None

            # Seat rank/return deltas
            seat_rank_delta = self.seat_rank_delta_head(seat_delta).squeeze(-1)
            seat_return_delta = self.seat_return_delta_head(seat_delta).squeeze(-1)

            # Get first seat→stock edge type for metadata computation
            seat_edges = None
            if seat_edge_index:
                for edges in seat_edge_index.values():
                    seat_edges = edges
                    break

            # Compute seat metadata and connection mask
            seat_metadata, has_seat_connection = self._compute_seat_metadata(
                torch.arange(base_embedding.size(0), device=base_embedding.device),
                seat_edges,
            )

            # Compute seat gate (only for stocks with seat connections)
            gate_input = torch.cat([base_embedding, seat_delta, seat_metadata], dim=-1)
            seat_gate = torch.sigmoid(self.seat_gate_mlp(gate_input).squeeze(-1))

            # Force gate to zero for stocks without seat connections
            seat_gate = seat_gate * has_seat_connection.float()
        else:
            # No seat branch - zero deltas, zero gate
            seat_rank_delta = torch.zeros_like(base_rank)
            seat_return_delta = torch.zeros_like(base_return)
            seat_gate = torch.zeros(base_embedding.size(0), device=base_embedding.device)
            seat_style_probs = None

        # ═══════════════════════════════════════════════════════════
        # Final Predictions
        # ═══════════════════════════════════════════════════════════
        final_rank = base_rank + seat_gate * seat_rank_delta
        final_return = base_return + seat_gate * seat_return_delta

        # Direction and uncertainty (using final embedding)
        direction_output = self.direction_head(base_embedding)
        if self.config.direction_classes == 2:
            direction_logit = direction_output.squeeze(-1)
        else:
            direction_logit = direction_output
        uncertainty = F.softplus(self.uncertainty_head(base_embedding).squeeze(-1))

        # Relation diagnostics for debugging
        relation_diagnostics: dict[str, Any] = {
            "has_base_edges": len(base_edge_index) > 0,
            "has_seat_edges": seat_edge_index is not None and len(seat_edge_index) > 0,
            "base_edge_count": sum(e.size(1) for e in base_edge_index.values()),
            "seat_edge_count": (
                sum(e.size(1) for e in seat_edge_index.values()) if seat_edge_index else 0
            ),
        }

        return {
            "base_rank_score": base_rank,
            "base_return_prediction": base_return,
            "seat_rank_delta": seat_rank_delta,
            "seat_return_delta": seat_return_delta,
            "seat_gate": seat_gate,
            "final_rank_score": final_rank,
            "final_return_prediction": final_return,
            "direction_logit": direction_logit,
            "direction_logits": direction_logit,
            "predicted_return": final_return,
            "uncertainty": uncertainty,
            "seat_style_probs": seat_style_probs,
            "relation_diagnostics": relation_diagnostics,
            "model_valid": False,
        }

    def parameter_count(self) -> int:
        """Count model parameters."""
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


def build_dt_seat_hgt_v3(
    config: DTSeatHGTv3Config,
    graph_schema: dict[str, Any] | None = None,
) -> DTSeatHGTv3Model:
    """Build DT-SeatHGT v3 model.

    Args:
        config: Model configuration
        graph_schema: Optional frozen schema for shape inference

    Returns:
        DTSeatHGTv3Model instance
    """
    model = DTSeatHGTv3Model(config, graph_schema)
    validation = model.validate_parameter_limit()
    if not validation["within_limit"]:
        raise ValueError(
            f"Parameter limit exceeded: {validation['parameter_count']} > "
            f"{validation['maximum_parameters']}"
        )
    return model
