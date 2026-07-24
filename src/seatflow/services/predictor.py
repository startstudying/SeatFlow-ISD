from __future__ import annotations

from math import tanh

from seatflow.graph.types import NodeType
from seatflow.models.contracts import ModelInput, PredictionResult


class HeuristicSmokePredictor:
    """Dependency-free predictor used only to validate the initialized data flow."""

    def __init__(self, horizons: tuple[int, ...]) -> None:
        self.horizons = horizons

    def predict(self, model_input: ModelInput) -> PredictionResult:
        stock_id = model_input.target_stock_id
        stock_edges = tuple(
            edge
            for edge in model_input.graph.edges
            if edge.target_type == NodeType.STOCK and edge.target_id == stock_id
        )
        if not stock_edges:
            raise ValueError(f"stock {stock_id!r} has no seat flow edges")

        buy_weight = sum(edge.weight for edge in stock_edges if edge.relation == "buys")
        sell_weight = sum(edge.weight for edge in stock_edges if edge.relation == "sells")
        gross_weight = buy_weight + sell_weight
        normalized_flow = (buy_weight - sell_weight) / gross_weight if gross_weight else 0.0
        rise_probability = 0.5 + 0.45 * tanh(2.0 * normalized_flow)
        expected_returns = {
            horizon: round(normalized_flow * min(horizon, 5) * 0.01, 6) for horizon in self.horizons
        }
        key_edges = sorted(stock_edges, key=lambda edge: edge.weight, reverse=True)[:3]
        key_seats = tuple(edge.source_id for edge in key_edges)
        paths = tuple((edge.source_id, stock_id) for edge in key_edges)

        risk_flags: list[str] = ["smoke_baseline_not_for_trading"]
        if gross_weight == 0:
            risk_flags.append("zero_gross_flow")
        if len(stock_edges) < 2:
            risk_flags.append("insufficient_seat_diversity")

        return PredictionResult(
            target_stock_id=stock_id,
            rise_probability=round(rise_probability, 6),
            expected_returns=expected_returns,
            key_seats=key_seats,
            propagation_paths=paths,
            risk_flags=tuple(risk_flags),
        )
