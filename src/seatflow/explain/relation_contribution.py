"""Relation contribution analysis for DT-SeatHGT v3.

This module provides tools for analyzing which relations contribute to predictions:
- Single edge masking: isolate individual Seat→Stock edge effects
- Seat-level masking: isolate all edges from a single seat
- Relation-level masking: isolate all Seat→Stock edges

NOTE: These tools do NOT modify the original graph. They create masked copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EdgeMaskingResult:
    """Result of edge masking analysis."""

    stock_id: str
    seat_id: str | None
    relation: str

    # Edge attributes (if available)
    edge_attr: list[float] | None = None
    direction: str | None = None  # "buys" or "sells"
    amount: float | None = None

    # Predictions
    prediction_before: float | None = None
    prediction_after: float | None = None

    # Delta
    prediction_delta: float = 0.0

    # Context
    seat_gate: float = 0.0
    identity_scope: str = "unknown"

    # Edge existence
    edge_existed: bool = True


@dataclass(frozen=True)
class SeatLevelMaskingResult:
    """Result of seat-level masking (all edges from one seat)."""

    seat_id: str

    # Number of edges masked
    num_edges_masked: int = 0

    # Stock IDs affected
    affected_stock_ids: list[str] | None = None

    # Aggregate prediction delta
    total_prediction_delta: float = 0.0
    mean_prediction_delta: float = 0.0
    max_prediction_delta: float = 0.0

    # Seat gate statistics
    seat_gate_before: float = 0.0
    seat_gate_after: float = 0.0

    # Identity scope
    identity_scope: str = "seat"


@dataclass(frozen=True)
class RelationMaskingResult:
    """Result of relation-level masking (all Seat→Stock edges)."""

    relation: str

    # Number of edges masked
    num_edges_masked: int = 0

    # Aggregate statistics
    stocks_affected: int = 0
    stocks_unaffected: int = 0

    # Prediction deltas
    mean_prediction_delta: float = 0.0
    max_prediction_delta: float = 0.0

    # Seat gate changes
    seat_gate_delta_mean: float = 0.0
    seat_gate_delta_max: float = 0.0


class RelationMasker:
    """Utilities for masking relations in heterogeneous graphs.

    All masking operations are non-destructive: they return new masked copies
    without modifying the original graph.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize with model for prediction computation."""
        self.model = model
        self.model.eval()

    def mask_edge_copy(
        self,
        graph_data: Any,
        edge_type: tuple[str, str, str],
        edge_idx: int,
    ) -> Any:
        """Create a copy of graph with one edge masked.

        Args:
            graph_data: Original HeteroData object
            edge_type: (source_type, relation_name, target_type)
            edge_idx: Index of edge to mask

        Returns:
            Copy of graph_data with edge masked (removed)
        """
        import copy

        # Deep copy the graph
        masked_graph = copy.deepcopy(graph_data)

        # Get edge store
        edge_store = masked_graph[edge_type]

        if not hasattr(edge_store, "edge_index"):
            return masked_graph

        # Get current edge index tensor
        edge_index = edge_store.edge_index

        if edge_index.shape[1] <= edge_idx:
            return masked_graph  # Edge doesn't exist

        # Create mask
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        mask[edge_idx] = False

        # Apply mask
        edge_store.edge_index = edge_index[:, mask]

        # Also mask edge_attr if present
        if hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None:
            edge_store.edge_attr = edge_store.edge_attr[mask]

        return masked_graph

    def mask_seat_edges_copy(
        self,
        graph_data: Any,
        seat_id: int,
    ) -> Any:
        """Create a copy of graph with all edges from a specific seat masked.

        Args:
            graph_data: Original HeteroData object
            seat_id: Index of seat to mask

        Returns:
            Copy of graph_data with all edges from this seat removed
        """
        import copy

        masked_graph = copy.deepcopy(graph_data)

        # Find all seat edge types
        seat_edge_types = [
            et for et in masked_graph.edge_types
            if et[0] == "seat"
        ]

        for edge_type in seat_edge_types:
            edge_store = masked_graph[edge_type]

            if not hasattr(edge_store, "edge_index"):
                continue

            edge_index = edge_store.edge_index

            if edge_index.shape[1] == 0:
                continue

            # Find edges where source is the seat
            source_mask = edge_index[0] != seat_id

            # Apply mask
            edge_store.edge_index = edge_index[:, source_mask]

            if hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None:
                edge_store.edge_attr = edge_store.edge_attr[source_mask]

        return masked_graph

    def mask_all_seat_edges_copy(self, graph_data: Any) -> Any:
        """Create a copy of graph with all Seat→Stock edges masked.

        Args:
            graph_data: Original HeteroData object

        Returns:
            Copy of graph_data with all seat edges removed
        """
        import copy

        masked_graph = copy.deepcopy(graph_data)

        # Find all seat edge types
        seat_edge_types = [
            et for et in masked_graph.edge_types
            if et[0] == "seat"
        ]

        for edge_type in seat_edge_types:
            edge_store = masked_graph[edge_type]

            # Remove all edges
            edge_store.edge_index = torch.empty((2, 0), dtype=torch.long)

            if hasattr(edge_store, "edge_attr"):
                # Keep attr dim for masked graph consistency
                edge_shape = edge_store.edge_attr.shape
                attr_dim = edge_shape[1] if len(edge_shape) > 1 else 1
                edge_store.edge_attr = torch.empty((0, attr_dim))

        return masked_graph

    def compute_prediction_with_mask(
        self,
        graph_data: Any,
        target_idx: int,
        enable_seat_branch: bool = True,
    ) -> tuple[float, float, dict[str, Any]]:
        """Compute prediction and seat gate for a target stock.

        Args:
            graph_data: HeteroData object
            target_idx: Index of target stock
            enable_seat_branch: Whether to enable seat branch

        Returns:
            Tuple of (final_return, seat_gate, full_outputs)
        """
        with torch.no_grad():
            outputs = self.model(graph_data, enable_seat_branch=enable_seat_branch)

        # Extract scalar values safely
        final_return = outputs["final_return_prediction"][target_idx]
        if final_return.dim() > 0:
            final_return = final_return.item()

        seat_gate = outputs["seat_gate"][target_idx]
        if seat_gate.dim() > 0:
            seat_gate = seat_gate.item()

        # Extract outputs dict, converting tensors to appropriate form
        def extract_value(v: Any, idx: int) -> Any:
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return v.item()
                elif v.dim() == 1 and idx < v.size(0):
                    return v[idx].item()
                elif v.dim() >= 2 and idx < v.size(0):
                    return v[idx].cpu().tolist()
                else:
                    return v.cpu().tolist()
            return v

        outputs_dict = {k: extract_value(v, target_idx) for k, v in outputs.items()}

        return (
            final_return,
            seat_gate,
            outputs_dict,
        )

    def analyze_single_edge(
        self,
        graph_data: Any,
        edge_type: tuple[str, str, str],
        edge_idx: int,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
    ) -> EdgeMaskingResult:
        """Analyze the contribution of a single edge.

        Args:
            graph_data: HeteroData object
            edge_type: (source_type, relation_name, target_type)
            edge_idx: Index of edge to analyze
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID

        Returns:
            EdgeMaskingResult with before/after predictions and delta
        """
        edge_store = graph_data[edge_type]

        if not hasattr(edge_store, "edge_index") or edge_store.edge_index.shape[1] <= edge_idx:
            return EdgeMaskingResult(
                stock_id=stock_id_map.get(edge_idx, "unknown") if stock_id_map else str(edge_idx),
                seat_id=seat_id_map.get(edge_idx, "unknown") if seat_id_map else None,
                relation="__".join(edge_type),
                edge_existed=False,
            )

        # Get edge info
        edge_index = edge_store.edge_index
        seat_idx = edge_index[0, edge_idx].item()
        stock_idx = edge_index[1, edge_idx].item()

        # Get edge attributes
        edge_attr = None
        direction = None
        amount = None
        if hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None:
            edge_attr = edge_store.edge_attr[edge_idx].cpu().numpy().tolist()
            # Try to parse common edge attributes
            if len(edge_attr) >= 1:
                direction = "buys" if edge_attr[0] > 0 else "sells"
            if len(edge_attr) >= 2:
                amount = float(edge_attr[1])

        # Compute before prediction
        pred_before, gate_before, _ = self.compute_prediction_with_mask(
            graph_data, stock_idx, enable_seat_branch=True
        )

        # Compute after prediction (with edge masked)
        masked_graph = self.mask_edge_copy(graph_data, edge_type, edge_idx)
        pred_after, gate_after, _ = self.compute_prediction_with_mask(
            masked_graph, stock_idx, enable_seat_branch=True
        )

        stock_id = stock_id_map.get(stock_idx, str(stock_idx)) if stock_id_map else str(stock_idx)
        seat_id = seat_id_map.get(seat_idx, str(seat_idx)) if seat_id_map else str(seat_idx)
        return EdgeMaskingResult(
            stock_id=stock_id,
            seat_id=seat_id,
            relation="__".join(edge_type),
            edge_attr=edge_attr,
            direction=direction,
            amount=amount,
            prediction_before=pred_before,
            prediction_after=pred_after,
            prediction_delta=pred_after - pred_before,
            seat_gate=gate_before,
            identity_scope="edge",
        )

    def analyze_seat_level(
        self,
        graph_data: Any,
        seat_idx: int,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
    ) -> SeatLevelMaskingResult:
        """Analyze the contribution of all edges from a single seat.

        Args:
            graph_data: HeteroData object
            seat_idx: Index of seat to analyze
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID

        Returns:
            SeatLevelMaskingResult with aggregate deltas
        """
        # Find all edges from this seat
        affected_stocks: list[int] = []

        seat_edge_types = [
            et for et in graph_data.edge_types
            if et[0] == "seat"
        ]

        for edge_type in seat_edge_types:
            edge_store = graph_data[edge_type]
            if not hasattr(edge_store, "edge_index"):
                continue

            edge_index = edge_store.edge_index
            source_mask = edge_index[0] == seat_idx
            affected = edge_index[1, source_mask].cpu().numpy().tolist()
            affected_stocks.extend(affected)

        if not affected_stocks:
            return SeatLevelMaskingResult(
                seat_id=seat_id_map.get(seat_idx, str(seat_idx)) if seat_id_map else str(seat_idx),
                num_edges_masked=0,
                affected_stock_ids=[],
                identity_scope="seat",
            )

        # Compute before predictions for all affected stocks
        pred_before_dict: dict[int, float] = {}
        gate_before_dict: dict[int, float] = {}
        for stock_idx in set(affected_stocks):
            pred_before_dict[stock_idx], gate_before_dict[stock_idx], _ = \
                self.compute_prediction_with_mask(graph_data, stock_idx, enable_seat_branch=True)

        # Compute after predictions (with all seat edges masked)
        masked_graph = self.mask_seat_edges_copy(graph_data, seat_idx)

        deltas: list[float] = []
        for stock_idx in set(affected_stocks):
            pred_after, gate_after, _ = self.compute_prediction_with_mask(
                masked_graph, stock_idx, enable_seat_branch=True
            )
            deltas.append(pred_before_dict[stock_idx] - pred_after)

        return SeatLevelMaskingResult(
            seat_id=seat_id_map.get(seat_idx, str(seat_idx)) if seat_id_map else str(seat_idx),
            num_edges_masked=len(affected_stocks),
            affected_stock_ids=[
                stock_id_map.get(s, str(s)) if stock_id_map else str(s)
                for s in set(affected_stocks)
            ],
            total_prediction_delta=sum(deltas),
            mean_prediction_delta=sum(deltas) / len(deltas) if deltas else 0.0,
            max_prediction_delta=max(deltas) if deltas else 0.0,
            seat_gate_before=(
                sum(gate_before_dict.values()) / len(gate_before_dict)
                if gate_before_dict else 0.0
            ),
            seat_gate_after=gate_after,
            identity_scope="seat",
        )

    def analyze_relation_level(
        self,
        graph_data: Any,
        stock_id_map: dict[int, str] | None = None,
    ) -> RelationMaskingResult:
        """Analyze the contribution of all Seat→Stock edges.

        Args:
            graph_data: HeteroData object
            stock_id_map: Optional mapping from index to stock ID

        Returns:
            RelationMaskingResult with aggregate statistics
        """
        # Count seat edges
        num_edges = 0
        seat_edge_types = [
            et for et in graph_data.edge_types
            if et[0] == "seat"
        ]

        for edge_type in seat_edge_types:
            edge_store = graph_data[edge_type]
            if hasattr(edge_store, "edge_index"):
                num_edges += edge_store.edge_index.shape[1]

        if num_edges == 0:
            return RelationMaskingResult(
                relation="seat__stock",
                num_edges_masked=0,
            )

        # Get number of stocks
        n_stocks = graph_data["stock"].x.shape[0]

        # Compute predictions before
        pred_before: list[float] = []
        gate_before: list[float] = []
        for stock_idx in range(n_stocks):
            pred, gate, _ = self.compute_prediction_with_mask(
                graph_data, stock_idx, enable_seat_branch=True
            )
            pred_before.append(pred)
            gate_before.append(gate)

        # Compute predictions after
        masked_graph = self.mask_all_seat_edges_copy(graph_data)

        pred_after: list[float] = []
        gate_after: list[float] = []
        for stock_idx in range(n_stocks):
            pred, gate, _ = self.compute_prediction_with_mask(
                masked_graph, stock_idx, enable_seat_branch=True
            )
            pred_after.append(pred)
            gate_after.append(gate)

        # Compute deltas
        deltas = [pb - pa for pb, pa in zip(pred_before, pred_after, strict=True)]

        return RelationMaskingResult(
            relation="seat__stock",
            num_edges_masked=num_edges,
            stocks_affected=sum(1 for d in deltas if abs(d) > 1e-6),
            stocks_unaffected=sum(1 for d in deltas if abs(d) <= 1e-6),
            mean_prediction_delta=sum(deltas) / len(deltas) if deltas else 0.0,
            max_prediction_delta=max(abs(d) for d in deltas) if deltas else 0.0,
            seat_gate_delta_mean=(
                sum(abs(gb - ga) for gb, ga in zip(gate_before, gate_after, strict=True)) / n_stocks
            ),
            seat_gate_delta_max=max(
                abs(gb - ga) for gb, ga in zip(gate_before, gate_after, strict=True)
            ),
        )


def explain_prediction_edge_contributions(
    model: nn.Module,
    graph_data: Any,
    target_index: int,
    stock_id_map: dict[int, str] | None = None,
    seat_id_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Generate comprehensive edge contribution explanation for a prediction.

    Args:
        model: Trained DT-SeatHGT v3 model
        graph_data: HeteroData object
        target_index: Index of target stock to explain
        stock_id_map: Optional mapping from index to stock ID
        seat_id_map: Optional mapping from index to seat ID

    Returns:
        Dictionary with edge-level attribution
    """
    masker = RelationMasker(model)

    # Find edges targeting this stock
    edge_contributions: list[dict[str, Any]] = []

    for edge_type in graph_data.edge_types:
        source_type, rel_name, target_type = edge_type

        if target_type != "stock":
            continue

        edge_store = graph_data[edge_type]
        if not hasattr(edge_store, "edge_index"):
            continue

        edge_index = edge_store.edge_index

        # Find edges targeting target_index
        for edge_idx in range(edge_index.shape[1]):
            if edge_index[1, edge_idx].item() == target_index:
                result = masker.analyze_single_edge(
                    graph_data, edge_type, edge_idx,
                    stock_id_map=stock_id_map,
                    seat_id_map=seat_id_map,
                )
                edge_contributions.append({
                    "seat_id": result.seat_id,
                    "relation": result.relation,
                    "direction": result.direction,
                    "amount": result.amount,
                    "prediction_delta": result.prediction_delta,
                    "seat_gate": result.seat_gate,
                })

    target_stock_id = (
        stock_id_map.get(target_index, str(target_index))
        if stock_id_map else str(target_index)
    )
    return {
        "target_index": target_index,
        "target_stock_id": target_stock_id,
        "edge_contributions": edge_contributions,
        "num_edges": len(edge_contributions),
        "total_delta": sum(e["prediction_delta"] for e in edge_contributions),
    }
