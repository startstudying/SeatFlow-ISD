from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seatflow.graph.tensorize import TensorizedGraph


class PyGModelUnavailableError(RuntimeError):
    pass


MODEL_ARCHITECTURE_VERSION = "dt_seat_hgt_v2_direction_schema"


@dataclass(frozen=True)
class DTSeatHGTConfig:
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    horizons: tuple[int, ...]
    target_node_type: str = "stock"
    direction_classes: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive")
        if self.direction_classes not in {2, 3}:
            raise ValueError("direction_classes must be 2 or 3")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["horizons"] = list(self.horizons)
        return payload


def require_pyg_model_stack() -> None:
    try:
        import torch  # noqa: F401
        from torch_geometric.nn import HGTConv  # noqa: F401
    except ModuleNotFoundError as error:
        raise PyGModelUnavailableError(
            "DTSeatHGT requires PyTorch and torch-geometric. "
            "Install the ml extra before running DT-SeatHGT training."
        ) from error


def build_dt_seat_hgt(config: DTSeatHGTConfig, tensorized: TensorizedGraph) -> Any:
    require_pyg_model_stack()
    return _build_model(config, tensorized)


def _build_model(config: DTSeatHGTConfig, tensorized: TensorizedGraph) -> Any:
    import torch
    from torch import nn
    from torch_geometric.nn import HGTConv

    class RelationAwareEdgeGate(nn.Module):
        def __init__(
            self,
            data: Any,
            metadata: tuple[list[str], list[tuple[str, str, str]]],
        ) -> None:
            super().__init__()
            self.expected_edge_attr_dims = {
                "::".join(edge_type): _edge_attr_dim(data[edge_type])
                for edge_type in metadata[1]
                if hasattr(data[edge_type], "edge_attr")
            }
            self.relation_gates = nn.ParameterDict(
                {
                    "::".join(edge_type): nn.Parameter(torch.ones(1))
                    for edge_type in metadata[1]
                }
            )
            self.edge_attr_encoders = nn.ModuleDict(
                {
                    relation: nn.Linear(edge_attr_dim, config.hidden_dim, bias=False)
                    for relation, edge_attr_dim in self.expected_edge_attr_dims.items()
                }
            )

        def forward(self, x_dict: dict[str, Any], data: Any) -> dict[str, Any]:
            gated = {node_type: features.clone() for node_type, features in x_dict.items()}
            for edge_type in data.edge_types:
                store = data[edge_type]
                if not hasattr(store, "edge_weight"):
                    raise ValueError(f"edge_weight missing for edge type {edge_type}")
                edge_index = store.edge_index
                edge_weight = store.edge_weight
                if edge_index.numel() == 0:
                    continue
                source_type, _, target_type = edge_type
                source_index = edge_index[0]
                target_index = edge_index[1]
                source_features = x_dict[source_type][source_index]
                weights = edge_weight.unsqueeze(-1)
                relation = "::".join(edge_type)
                messages = source_features * weights * self.relation_gates[relation]
                if relation in self.expected_edge_attr_dims:
                    edge_attr = _validated_edge_attr(
                        store,
                        edge_type,
                        self.expected_edge_attr_dims[relation],
                    )
                    messages = messages + self.edge_attr_encoders[relation](edge_attr)
                aggregate = torch.zeros_like(gated[target_type])
                aggregate.index_add_(0, target_index, messages)
                degree = torch.zeros(
                    (gated[target_type].size(0), 1),
                    dtype=gated[target_type].dtype,
                    device=gated[target_type].device,
                )
                degree.index_add_(0, target_index, torch.ones_like(weights))
                gated[target_type] = gated[target_type] + aggregate / degree.clamp_min(1.0)
            return gated

    class DTSeatHGT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            data = tensorized.data
            metadata = data.metadata()
            in_channels = {node_type: int(data[node_type].x.size(-1)) for node_type in metadata[0]}
            self.input_projection = nn.ModuleDict(
                {
                    node_type: nn.Linear(width, config.hidden_dim)
                    for node_type, width in in_channels.items()
                }
            )
            self.layers = nn.ModuleList(
                [
                    HGTConv(
                        in_channels=config.hidden_dim,
                        out_channels=config.hidden_dim,
                        metadata=metadata,
                        heads=config.num_heads,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.edge_weight_gate = RelationAwareEdgeGate(data, metadata)
            self.dropout = nn.Dropout(config.dropout)
            self.direction_heads = nn.ModuleDict(
                {
                    str(horizon): nn.Linear(
                        config.hidden_dim, 1 if config.direction_classes == 2 else 3
                    )
                    for horizon in config.horizons
                }
            )
            self.return_heads = nn.ModuleDict(
                {str(horizon): nn.Linear(config.hidden_dim, 1) for horizon in config.horizons}
            )

        def forward(self, data: Any) -> dict[str, object]:
            if config.target_node_type not in data.node_types:
                raise ValueError(f"target node type {config.target_node_type!r} is missing")
            x_dict = {
                node_type: self.input_projection[node_type](data[node_type].x)
                for node_type in data.node_types
            }
            for layer in self.layers:
                weighted_x_dict = self.edge_weight_gate(x_dict, data)
                updated = layer(weighted_x_dict, data.edge_index_dict)
                x_dict = {
                    node_type: self.dropout(torch.relu(updated.get(node_type, features)))
                    for node_type, features in weighted_x_dict.items()
                }
            target = x_dict.get(config.target_node_type)
            if target is None or int(target.size(0)) == 0:
                raise ValueError("target stock nodes are required for DTSeatHGT forward")
            return {
                "direction_logits": {
                    horizon: self.direction_heads[str(horizon)](target).squeeze(-1)
                    for horizon in config.horizons
                },
                "return_predictions": {
                    horizon: self.return_heads[str(horizon)](target).squeeze(-1)
                    for horizon in config.horizons
                },
                "target_node_type": config.target_node_type,
                "model_valid": False,
            }

    return DTSeatHGT()


def dt_seat_hgt_checkpoint_payload(config: DTSeatHGTConfig) -> dict[str, object]:
    return {
        "model_name": "DTSeatHGT",
        "architecture_version": MODEL_ARCHITECTURE_VERSION,
        "architecture": (
            "relation_aware_edge_weight_and_edge_attr_gate + "
            "HGTConv + multi_task_direction_and_return_heads"
        ),
        "config": config.to_dict(),
        "model_valid": False,
        "risk_flags": [
            "no_real_training_or_backtest",
            "not_model_valid_without_real_protocol_evidence",
            "not_for_research_claims_or_trading",
        ],
    }


def _edge_attr_dim(edge_store: Any) -> int:
    edge_attr = edge_store.edge_attr
    if int(edge_attr.dim()) != 2:
        raise ValueError("edge_attr must be a 2D tensor")
    edge_attr_dim = int(edge_attr.size(-1))
    if edge_attr_dim <= 0:
        raise ValueError("edge_attr width must be positive")
    return edge_attr_dim


def _validated_edge_attr(
    edge_store: Any,
    edge_type: tuple[str, str, str],
    expected_dim: int,
) -> Any:
    if not hasattr(edge_store, "edge_attr"):
        raise ValueError(f"edge_attr missing for edge type {edge_type}")
    edge_attr = edge_store.edge_attr
    if int(edge_attr.dim()) != 2:
        raise ValueError(f"edge_attr for edge type {edge_type} must be a 2D tensor")
    if int(edge_attr.size(0)) != int(edge_store.edge_index.size(1)):
        raise ValueError(f"edge_attr row count mismatch for edge type {edge_type}")
    if int(edge_attr.size(-1)) != expected_dim:
        raise ValueError(
            f"edge_attr width mismatch for edge type {edge_type}: "
            f"expected {expected_dim}, got {int(edge_attr.size(-1))}"
        )
    return edge_attr
