"""Typed Edge-Gated Relation Layer for heterogeneous graph propagation.

This module provides a proper heterogeneous graph layer that:
- Processes edges by relation type independently
- Uses edge attributes for gating
- Supports relation-specific Q/K/V projections
- Normalizes attention by target node in-degree
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    pass


def _log_softmax_per_dst(
    src: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Compute log-softmax normalized by destination node.

    Numerically stable log-softmax:
    log_softmax_i = src_i - max_dst - log(sum_j(exp(src_j - max_dst)))

    For edges pointing to the same destination, the result exp(log_softmax)
    sums to 1.0 per destination.

    Args:
        src: [num_edges] logit values per edge
        index: [num_edges] destination node index for each edge
        num_nodes: total number of destination nodes

    Returns:
        [num_edges] log-softmax values per edge
    """
    # Step 1: Get max logit per destination node for numerical stability
    max_per_dst = torch.full((num_nodes,), float('-inf'), device=src.device, dtype=src.dtype)
    max_per_dst.scatter_reduce_(0, index, src, reduce='amax', include_self=False)

    # Step 2: Subtract max per destination (src_stable <= 0 for all edges)
    src_stable = src - max_per_dst[index]

    # Step 3: Exp
    exp_src = src_stable.exp()

    # Step 4: Sum exp per destination node
    sum_exp = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    sum_exp.scatter_add_(0, index, exp_src)

    # Step 5: log_sum_exp = log(sum(exp(src - max))) = log_sum_exp
    log_sum_exp = sum_exp.clamp(min=1e-8).log()

    # Step 6: log_softmax = src_stable - log_sum_exp[dest]
    # (max cancels out: src - max - log_sum_exp)
    return src_stable - log_sum_exp[index]


@dataclass(frozen=True)
class RelationConfig:
    """Configuration for a single relation type."""

    source_type: str
    target_type: str
    relation_name: str
    query_dim: int
    key_dim: int
    value_dim: int
    out_dim: int


class TypedEdgeGatedRelationLayer(nn.Module):
    """Heterogeneous relation propagation with typed edges and edge gating.

    Processes each relation type independently with:
    - Relation-specific Q/K/V projections
    - Edge attribute encoder
    - Multi-head attention (num_heads configurable)
    - Edge-gated attention normalized by target in-degree
    - Residual connection with LayerNorm

    Args:
        relation_configs: Dict mapping relation_name -> RelationConfig
        edge_attr_dims: Dict mapping relation_name -> edge attribute dimension
        num_heads: Number of attention heads for multi-head attention
        dropout: Dropout probability
        use_edge_gate: Whether to use edge attributes for gating
    """

    def __init__(
        self,
        relation_configs: dict[str, RelationConfig],
        edge_attr_dims: dict[str, int],
        num_heads: int = 1,
        dropout: float = 0.1,
        use_edge_gate: bool = True,
    ) -> None:
        super().__init__()

        self.relation_configs = relation_configs
        self.num_heads = num_heads
        self.use_edge_gate = use_edge_gate

        # Per-relation projection modules (use string keys for ModuleDict)
        self.query_proj: nn.ModuleDict = nn.ModuleDict()
        self.key_proj: nn.ModuleDict = nn.ModuleDict()
        self.value_proj: nn.ModuleDict = nn.ModuleDict()
        self.edge_encoder: nn.ModuleDict = nn.ModuleDict()
        self.relation_scale: nn.ParameterDict = nn.ParameterDict()

        for rel_name, config in relation_configs.items():
            # Q/K/V projections
            self.query_proj[rel_name] = nn.Linear(config.query_dim, config.out_dim)
            self.key_proj[rel_name] = nn.Linear(config.key_dim, config.out_dim)
            self.value_proj[rel_name] = nn.Linear(config.value_dim, config.out_dim)

            # Edge attribute encoder (if edge attrs exist)
            edge_dim = edge_attr_dims.get(rel_name, 0)
            if edge_dim > 0 and use_edge_gate:
                self.edge_encoder[rel_name] = nn.Sequential(
                    nn.Linear(edge_dim, config.out_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(config.out_dim, 1),
                )

            # Learnable relation importance scale
            self.relation_scale[rel_name] = nn.Parameter(torch.zeros(1))

        # Layer norm (shared; all relations use the same out_dim == hidden_dim)
        # and dropout. No projection is created lazily in forward() (V4-A4).
        self.layer_norm = nn.LayerNorm(relation_configs[next(iter(relation_configs))].out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
        edge_attr_dict: dict[str, torch.Tensor] | None = None,
        allowed_edge_types: set[str] | None = None,
        zero_identity_for_unconnected: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Propagate information through typed edges.

        Args:
            x_dict: Dict mapping node_type -> [N_nodes, feature_dim] tensors
            edge_index_dict: Dict mapping relation_name -> [2, num_edges] tensors
            edge_attr_dict: Optional dict mapping relation_name -> [num_edges, attr_dim]
            allowed_edge_types: Optional set of relation names to process
            zero_identity_for_unconnected: If True, the branch returns a pure
                message-passing *delta*: connected nodes receive the normalized
                aggregated message (LayerNorm + dropout) while nodes without any
                incoming edge receive zeros. The caller is responsible for adding
                the delta to its own residual. When False, returns the updated
                embedding ``LayerNorm(dropout(msg) + x)``.

        Returns:
            Dict mapping node_type -> tensor. Shape is ``[N_nodes, out_dim]``.
        """
        # aggregated_msg[node_type] holds the per-node message-passing output
        # (summed across relations of the same target type). has_edge[node_type]
        # is a boolean mask of which nodes received at least one incoming edge.
        aggregated_msg: dict[str, torch.Tensor] = {}
        has_edge_mask: dict[str, torch.Tensor] = {}

        for rel_name, edge_index in edge_index_dict.items():
            # Skip if relation not in our configs
            if rel_name not in self.relation_configs:
                continue

            # Check if this relation is allowed
            if allowed_edge_types is not None and rel_name not in allowed_edge_types:
                continue

            config = self.relation_configs[rel_name]

            # Get source and target node features
            src_features = x_dict.get(config.source_type)
            dst_features = x_dict.get(config.target_type)

            if src_features is None or dst_features is None:
                continue

            # Skip empty edge sets (avoid zero-length indexing/slicing)
            if edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            # Project to Q/K/V
            query = self.query_proj[rel_name](dst_features)  # [N_dst, out_dim]
            key = self.key_proj[rel_name](src_features)  # [N_src, out_dim]
            value = self.value_proj[rel_name](src_features)  # [N_src, out_dim]

            # Get edges
            src_idx = edge_index[0]  # [num_edges]
            dst_idx = edge_index[1]  # [num_edges]
            num_edges = src_idx.size(0)

            # V4-A2: True multi-head attention. Keep the per-head structure all
            # the way through aggregation instead of collapsing heads early.
            head_dim = config.out_dim // self.num_heads

            # Index Q/K/V by edge endpoints and reshape for multi-head.
            q = query[dst_idx].view(num_edges, self.num_heads, head_dim)
            k = key[src_idx].view(num_edges, self.num_heads, head_dim)
            v = value[src_idx].view(num_edges, self.num_heads, head_dim)

            # Per-head attention logits: [num_edges, num_heads]
            attention_logits = (q * k).sum(dim=-1) / (head_dim ** 0.5)

            # Compute edge gate if available. The gate modulates the *message*
            # magnitude (value * attention_weight * gate) rather than the
            # pre-softmax logits, so it affects single-in-edge destinations too.
            # Scaling logits before softmax would be a no-op for destinations with
            # exactly one incoming edge (softmax of a single element is always 1).
            has_edge_gate = (
                self.use_edge_gate
                and edge_attr_dict is not None
                and rel_name in edge_attr_dict
                and rel_name in self.edge_encoder
            )
            edge_gate: torch.Tensor | None = None
            if has_edge_gate and edge_attr_dict is not None:
                edge_attrs = edge_attr_dict[rel_name]
                edge_gate = torch.sigmoid(
                    self.edge_encoder[rel_name](edge_attrs).squeeze(-1)
                )  # [num_edges] in (0, 1)

            # Scale by relation importance (broadcasts across heads)
            attention_logits = attention_logits * (1 + self.relation_scale[rel_name])

            # V4-A1: Proper attention normalization. Compute a numerically stable
            # log-softmax over the incoming edges of each destination node, per
            # head. The *weights* used to aggregate values are exp(log-softmax)
            # (i.e. the softmax probabilities), so they sum to 1 per destination
            # per head. Using the log-softmax values directly would zero-out the
            # aggregation for nodes with a single incoming edge (log(1) == 0).
            num_dst_nodes = dst_features.size(0)
            log_attn = torch.zeros_like(attention_logits)  # [num_edges, num_heads]
            for h in range(self.num_heads):
                log_attn[:, h] = _log_softmax_per_dst(
                    attention_logits[:, h], dst_idx, num_dst_nodes
                )
            attn_weights = log_attn.exp()  # [num_edges, num_heads], sums to 1 per dst

            # Softmax-normalized value aggregation per head, then modulate the
            # message by the edge gate so the gate controls message strength
            # independent of in-edge count.
            # v: [num_edges, num_heads, head_dim], attn_weights: [num_edges, num_heads, 1]
            value_agg_per_head = v * attn_weights.unsqueeze(-1)
            if edge_gate is not None:
                # Broadcast gate across heads: [num_edges] -> [num_edges, num_heads, 1]
                value_agg_per_head = value_agg_per_head * edge_gate.view(
                    num_edges, 1, 1
                )

            # Aggregate values per destination node per head.
            aggregated_per_head = torch.zeros(
                num_dst_nodes, self.num_heads, head_dim, device=dst_features.device
            )
            dst_idx_expanded = dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(
                value_agg_per_head
            )
            aggregated_per_head.scatter_add_(0, dst_idx_expanded, value_agg_per_head)

            # Concatenate heads: [N_dst, num_heads, head_dim] -> [N_dst, out_dim]
            aggregated = aggregated_per_head.reshape(
                num_dst_nodes, self.num_heads * head_dim
            )

            dst_type = config.target_type

            # Track which destination nodes received at least one edge.
            edge_count = torch.zeros(num_dst_nodes, device=dst_features.device)
            edge_count.scatter_add_(
                0, dst_idx, torch.ones(num_edges, device=dst_features.device)
            )
            connected = edge_count > 0

            if dst_type in aggregated_msg:
                aggregated_msg[dst_type] = aggregated_msg[dst_type] + aggregated
                has_edge_mask[dst_type] = has_edge_mask[dst_type] | connected
            else:
                aggregated_msg[dst_type] = aggregated
                has_edge_mask[dst_type] = connected

        # Build the final output per node type.
        output_dict: dict[str, torch.Tensor] = {}
        for node_type, msg in aggregated_msg.items():
            if node_type not in x_dict:
                # Target type has no input features; nothing to residualize against.
                output_dict[node_type] = msg
                continue

            features = x_dict[node_type]
            out_dim = msg.size(-1)
            in_dim = features.size(-1)

            # All branches in this codebase use out_dim == in_dim (hidden_dim).
            # If a future caller needs a projection, it must be registered in
            # __init__; we never create nn.Linear inside forward (V4-A4).
            if out_dim != in_dim:
                raise ValueError(
                    f"Relation output dim {out_dim} != input dim {in_dim} for "
                    f"node type '{node_type}'. Register a projection in __init__."
                )

            if zero_identity_for_unconnected:
                # V4-A6: Branch returns a pure delta. Connected nodes get the
                # LayerNorm/dropout-stabilized message; unconnected nodes get a
                # hard zero so seat/base information never broadcasts to nodes
                # that have no edge in this relation.
                edge_mask: torch.Tensor | None = has_edge_mask.get(node_type)
                if edge_mask is None:
                    delta = torch.zeros_like(features)
                else:
                    delta = self.layer_norm(self.dropout(msg)) * edge_mask.unsqueeze(
                        -1
                    ).to(msg.dtype)
                output_dict[node_type] = delta
            else:
                # Updated-embedding semantics: LayerNorm(dropout(msg) + x).
                output_dict[node_type] = self.layer_norm(
                    self.dropout(msg) + features
                )

        # Handle node types present in x_dict that received no edges at all.
        for node_type, features in x_dict.items():
            if node_type in output_dict:
                continue
            if zero_identity_for_unconnected:
                # No incoming edges of any handled relation -> zero delta.
                output_dict[node_type] = torch.zeros_like(features)
            else:
                output_dict[node_type] = self.layer_norm(features)

        return output_dict

    def get_relation_names(self) -> list[str]:
        """Return list of relation names handled by this layer."""
        return list(self.relation_configs.keys())

    def get_relation_diagnostics(self) -> dict[str, dict[str, Any]]:
        """Return diagnostics for each relation type.

        Returns:
            Dict mapping relation name -> diagnostic info (scales, edge gate config, etc.)
        """
        diagnostics: dict[str, dict[str, Any]] = {}

        for rel_name, config in self.relation_configs.items():
            diag: dict[str, Any] = {
                "source_type": config.source_type,
                "target_type": config.target_type,
                "has_edge_encoder": rel_name in self.edge_encoder,
                "relation_scale": self.relation_scale[rel_name].item(),
            }
            diagnostics[rel_name] = diag

        return diagnostics


class BaseRelationBranch(nn.Module):
    """Base relation branch: Industry→Stock, Stock→Stock.

    Isolated from seat relations to prevent cross-contamination.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Define allowed base relations (use string keys for ModuleDict compatibility)
        base_relations = {
            "industry_to_stock": RelationConfig(
                source_type="industry",
                target_type="stock",
                relation_name="industry_to_stock",
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                out_dim=hidden_dim,
            ),
            "stock_to_stock": RelationConfig(
                source_type="stock",
                target_type="stock",
                relation_name="stock_to_stock",
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                out_dim=hidden_dim,
            ),
        }

        edge_attr_dims = {
            "industry_to_stock": 0,  # No edge attributes
            "stock_to_stock": 0,
        }

        self.layer = TypedEdgeGatedRelationLayer(
            relation_configs=base_relations,
            edge_attr_dims=edge_attr_dims,
            num_heads=num_heads,  # Pass num_heads for true multi-head attention
            dropout=dropout,
            use_edge_gate=False,
        )

        # Map from relation name to edge type tuple
        self._relation_to_edge_type = {
            "industry_to_stock": ("industry", "industry_to_stock", "stock"),
            "stock_to_stock": ("stock", "stock_to_stock", "stock"),
        }
        self._edge_type_to_relation = {v: k for k, v in self._relation_to_edge_type.items()}

    @property
    def allowed_edge_types(self) -> set[str]:
        """Return set of allowed relation names (for backward compat with tuple-based code)."""
        return set(self._relation_to_edge_type.keys())

    @property
    def allowed_edge_type_tuples(self) -> set[tuple[str, str, str]]:
        """Return set of allowed edge type tuples (src, relation, dst)."""
        return set(self._relation_to_edge_type.values())

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process base relations only.

        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by relation (tuple keys)
            edge_attr_dict: Edge attributes by relation (tuple keys)

        Returns:
            Updated node features
        """
        # Convert tuple keys to string keys for the layer
        string_keyed_edge_index: dict[str, torch.Tensor] = {}
        string_keyed_edge_attr: dict[str, torch.Tensor] | None = None

        for edge_type, edges in edge_index_dict.items():
            rel_name = self._edge_type_to_relation.get(edge_type)
            if rel_name:
                string_keyed_edge_index[rel_name] = edges

        if edge_attr_dict:
            string_keyed_edge_attr = {}
            for edge_type, attrs in edge_attr_dict.items():
                rel_name = self._edge_type_to_relation.get(edge_type)
                if rel_name:
                    string_keyed_edge_attr[rel_name] = attrs

        result: dict[str, torch.Tensor] = self.layer(
            x_dict=x_dict,
            edge_index_dict=string_keyed_edge_index,
            edge_attr_dict=string_keyed_edge_attr,
            zero_identity_for_unconnected=True,  # Return zeros for unconnected stocks
        )
        return result


class SeatRelationBranch(nn.Module):
    """Seat relation branch: Seat→Stock buys/sells.

    Isolated from base relations to prevent cross-contamination.
    Only processes Seat→Stock edges.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_edge_gate: bool = True,
        edge_attr_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        # Define allowed seat relations (use string keys)
        seat_relations = {
            "seat_buy_stock": RelationConfig(
                source_type="seat",
                target_type="stock",
                relation_name="seat_buy_stock",
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                out_dim=hidden_dim,
            ),
            "seat_sell_stock": RelationConfig(
                source_type="seat",
                target_type="stock",
                relation_name="seat_sell_stock",
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                out_dim=hidden_dim,
            ),
        }

        # Default edge attr dims if not provided
        default_edge_attr_dims = {
            "seat_buy_stock": 7,
            "seat_sell_stock": 7,
        }
        effective_edge_attr_dims = (
            edge_attr_dims if edge_attr_dims is not None else default_edge_attr_dims
        )

        self.layer = TypedEdgeGatedRelationLayer(
            relation_configs=seat_relations,
            edge_attr_dims=effective_edge_attr_dims,
            num_heads=num_heads,  # Pass num_heads for true multi-head attention
            dropout=dropout,
            use_edge_gate=use_edge_gate,
        )

        # Map from relation name to edge type tuple
        self._relation_to_edge_type = {
            "seat_buy_stock": ("seat", "seat_buy_stock", "stock"),
            "seat_sell_stock": ("seat", "seat_sell_stock", "stock"),
        }
        self._edge_type_to_relation = {v: k for k, v in self._relation_to_edge_type.items()}

    @property
    def allowed_edge_types(self) -> set[str]:
        """Return set of allowed relation names (for backward compat with tuple-based code)."""
        return set(self._relation_to_edge_type.keys())

    @property
    def allowed_edge_type_tuples(self) -> set[tuple[str, str, str]]:
        """Return set of allowed edge type tuples (src, relation, dst)."""
        return set(self._relation_to_edge_type.values())

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process seat relations only.

        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by relation (tuple keys)
            edge_attr_dict: Edge attributes by relation (tuple keys)

        Returns:
            Updated node features for stock nodes
        """
        # Convert tuple keys to string keys for the layer
        string_keyed_edge_index: dict[str, torch.Tensor] = {}
        string_keyed_edge_attr: dict[str, torch.Tensor] | None = None

        for edge_type, edges in edge_index_dict.items():
            rel_name = self._edge_type_to_relation.get(edge_type)
            if rel_name:
                string_keyed_edge_index[rel_name] = edges

        if edge_attr_dict:
            string_keyed_edge_attr = {}
            for edge_type, attrs in edge_attr_dict.items():
                rel_name = self._edge_type_to_relation.get(edge_type)
                if rel_name:
                    string_keyed_edge_attr[rel_name] = attrs

        result: dict[str, torch.Tensor] = self.layer(
            x_dict=x_dict,
            edge_index_dict=string_keyed_edge_index,
            edge_attr_dict=string_keyed_edge_attr,
            zero_identity_for_unconnected=True,  # Return zeros for unconnected stocks
        )
        return result
