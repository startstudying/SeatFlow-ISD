"""Memory-bounded heterogeneous ranker for SeatFlow R2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

from seatflow.graph.tensorize import TensorizedGraph


class SeatRankLiteUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeatRankLiteConfig:
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    edge_hidden_dim: int = 16
    dropout: float = 0.15
    relation_dropout: float = 0.10
    use_residual: bool = True
    use_layer_norm: bool = True
    target_node_type: str = "stock"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim >= 128:
            raise ValueError("hidden_dim must be in [1, 127]")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("num_heads must evenly divide hidden_dim")
        if self.num_layers not in {1, 2}:
            raise ValueError("SeatRankLite supports one or two layers")
        if self.edge_hidden_dim <= 0:
            raise ValueError("edge_hidden_dim must be positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.relation_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_seat_rank_lite(config: SeatRankLiteConfig, tensorized: TensorizedGraph) -> Any:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as error:
        raise SeatRankLiteUnavailableError(
            "SeatRankLite requires PyTorch and torch-geometric"
        ) from error
    model = _build_model(config, tensorized)
    validate_parameter_count(model)
    return model


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def validate_parameter_count(
    model: Any,
    *,
    recommended_max: int = 3_000_000,
    hard_max: int = 5_000_000,
) -> dict[str, int | bool]:
    if recommended_max <= 0 or hard_max < recommended_max:
        raise ValueError("parameter limits require 0 < recommended_max <= hard_max")
    count = count_parameters(model)
    if count > hard_max:
        raise ValueError(f"SeatRankLite parameter hard limit exceeded: {count} > {hard_max}")
    return {
        "parameter_count": count,
        "recommended_max": recommended_max,
        "hard_max": hard_max,
        "recommended_limit_exceeded": count > recommended_max,
    }


def _build_model(config: SeatRankLiteConfig, tensorized: TensorizedGraph) -> Any:
    import torch
    from torch import nn

    data = tensorized.data
    node_types, edge_types = data.metadata()
    edge_attr_dims = {
        _relation_key(edge_type): int(data[edge_type].edge_attr.size(-1))
        for edge_type in edge_types
    }
    head_dim = config.hidden_dim // config.num_heads

    class EdgeGatedHGTLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.ModuleDict(
                {
                    node_type: nn.Linear(config.hidden_dim, config.hidden_dim)
                    for node_type in node_types
                }
            )
            self.key = nn.ModuleDict(
                {
                    node_type: nn.Linear(config.hidden_dim, config.hidden_dim)
                    for node_type in node_types
                }
            )
            self.value = nn.ModuleDict(
                {
                    node_type: nn.Linear(config.hidden_dim, config.hidden_dim)
                    for node_type in node_types
                }
            )
            self.output = nn.ModuleDict(
                {
                    node_type: nn.Linear(config.hidden_dim, config.hidden_dim)
                    for node_type in node_types
                }
            )
            self.edge_gates = nn.ModuleDict(
                {
                    relation: nn.Sequential(
                        nn.Linear(width, config.edge_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(config.edge_hidden_dim, config.num_heads),
                    )
                    for relation, width in edge_attr_dims.items()
                }
            )
            self.relation_scale = nn.ParameterDict(
                {
                    _relation_key(edge_type): nn.Parameter(torch.ones(config.num_heads))
                    for edge_type in edge_types
                }
            )
            self.gate_dropout = nn.Dropout(config.relation_dropout)
            self.last_edge_gate_shapes: dict[str, tuple[int, int]] = {}

        def forward(self, x_dict: dict[str, Any], graph: Any) -> dict[str, Any]:
            aggregates = {
                node_type: torch.zeros_like(values) for node_type, values in x_dict.items()
            }
            degrees = {
                node_type: torch.zeros(
                    (values.size(0), 1), device=values.device, dtype=values.dtype
                )
                for node_type, values in x_dict.items()
            }
            for edge_type in graph.edge_types:
                source_type, _, target_type = edge_type
                store = graph[edge_type]
                if store.edge_index.numel() == 0:
                    continue
                source_index, target_index = store.edge_index
                query = self.query[target_type](x_dict[target_type]).index_select(
                    0, target_index
                ).view(-1, config.num_heads, head_dim)
                key = self.key[source_type](x_dict[source_type]).index_select(
                    0, source_index
                ).view(-1, config.num_heads, head_dim)
                value = self.value[source_type](x_dict[source_type]).index_select(
                    0, source_index
                ).view(-1, config.num_heads, head_dim)
                relation = _relation_key(edge_type)
                edge_gate = torch.sigmoid(self.edge_gates[relation](store.edge_attr))
                expected_shape = (int(store.edge_index.size(1)), config.num_heads)
                if tuple(edge_gate.shape) != expected_shape:
                    raise ValueError(
                        f"edge gate shape for {edge_type} must be {expected_shape}, "
                        f"got {tuple(edge_gate.shape)}"
                    )
                self.last_edge_gate_shapes[relation] = expected_shape
                edge_gate = self.gate_dropout(edge_gate)
                attention = torch.sigmoid(
                    (query * key).sum(dim=-1) / sqrt(head_dim)
                    * self.relation_scale[relation].to(dtype=query.dtype)
                )
                messages = (
                    value * (attention * edge_gate).unsqueeze(-1)
                ).to(dtype=aggregates[target_type].dtype)
                aggregates[target_type].index_add_(
                    0, target_index, messages.reshape(-1, config.hidden_dim)
                )
                degrees[target_type].index_add_(
                    0,
                    target_index,
                    torch.ones(
                        (target_index.numel(), 1),
                        device=target_index.device,
                        dtype=x_dict[target_type].dtype,
                    ),
                )
            return {
                node_type: self.output[node_type](
                    aggregates[node_type] / degrees[node_type].clamp_min(1.0)
                )
                for node_type in node_types
            }

    class SeatRankLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_projection = nn.ModuleDict(
                {
                    node_type: nn.Linear(int(data[node_type].x.size(-1)), config.hidden_dim)
                    for node_type in node_types
                }
            )
            self.layers = nn.ModuleList(EdgeGatedHGTLayer() for _ in range(config.num_layers))
            self.norms = nn.ModuleList(
                nn.ModuleDict(
                    {
                        node_type: (
                            nn.LayerNorm(config.hidden_dim)
                            if config.use_layer_norm
                            else nn.Identity()
                        )
                        for node_type in node_types
                    }
                )
                for _ in range(config.num_layers)
            )
            self.dropout = nn.Dropout(config.dropout)
            self.rank_head = nn.Linear(config.hidden_dim, 1)
            self.residual_return_head = nn.Linear(config.hidden_dim, 1)
            self.direction_head = nn.Linear(config.hidden_dim, 1)
            self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

        def forward(self, graph: Any) -> dict[str, Any]:
            x_dict = {
                node_type: self.input_projection[node_type](graph[node_type].x)
                for node_type in node_types
            }
            for layer, norms in zip(self.layers, self.norms, strict=True):
                updated = layer(x_dict, graph)
                x_dict = {
                    node_type: norms[node_type](
                        self.dropout(
                            torch.relu(
                                updated[node_type]
                                + (x_dict[node_type] if config.use_residual else 0.0)
                            )
                        )
                    )
                    for node_type in node_types
                }
            stock = x_dict.get(config.target_node_type)
            if stock is None or stock.size(0) == 0:
                raise ValueError("SeatRankLite requires target stock nodes")
            return {
                "rank_score": self.rank_head(stock).squeeze(-1),
                "residual_return": self.residual_return_head(stock).squeeze(-1),
                "direction_logit": self.direction_head(stock).squeeze(-1),
                "uncertainty": torch.nn.functional.softplus(
                    self.uncertainty_head(stock).squeeze(-1)
                ),
                "model_valid": False,
            }

    return SeatRankLite()


def _relation_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)
