"""Explainability module for DT-SeatHGT v3 model.

This module provides tools for interpreting model predictions, including:
- Component contribution analysis (base vs seat contributions)
- Gate activation analysis
- Edge masking for critical subgraph identification
- Relation contribution analysis

NOTE: The model uses GRU for temporal encoding, not attention.
The GRU processes sequences to produce a single hidden state.
Use RelationMasker and CriticalSubgraphAnalyzer for edge-level attribution.

Modules:
- relation_contribution: Edge-level masking and attribution
- critical_subgraph: Critical subgraph identification

Legacy components (kept for diagnostics):
- analyze_component_contributions
- analyze_gate_activations
- explain_prediction (component-level)
- generate_explainability_report
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from seatflow.explain.critical_subgraph import (
    CriticalSubgraphAnalyzer,
    CriticalSubgraphResult,
    SeatInfluenceCluster,
    summarize_critical_subgraph,
)

# Import new modules
from seatflow.explain.relation_contribution import (
    EdgeMaskingResult,
    RelationMasker,
    RelationMaskingResult,
    SeatLevelMaskingResult,
    explain_prediction_edge_contributions,
)
from seatflow.models.dt_seat_hgt_v3 import DTSeatHGTv3Model


@dataclass(frozen=True)
class ComponentContribution:
    """Analysis of how base vs seat branches contribute to predictions."""

    base_rank_score: float
    seat_rank_delta: float
    seat_gate_value: float
    final_rank_score: float

    seat_contribution_ratio: float  # How much seat delta modified base score
    gate_activation_ratio: float  # Normalized gate value (0-1)

    base_return: float
    seat_return_delta: float
    final_return: float


@dataclass(frozen=True)
class GateAnalysis:
    """Analysis of seat gate activations across stocks."""

    gate_values: list[float]
    gate_mean: float
    gate_std: float
    gate_min: float
    gate_max: float

    high_activation_count: int  # Count where gate > 0.5
    low_activation_count: int  # Count where gate <= 0.5

    interpretation: str  # Human-readable summary


def analyze_component_contributions(
    model: DTSeatHGTv3Model,
    graph_data: Any,
    target_indices: list[int] | None = None,
) -> list[ComponentContribution]:
    """Analyze how base and seat components contribute to predictions.

    Args:
        model: Trained DT-SeatHGT v3 model
        graph_data: PyG HeteroData graph
        target_indices: Specific stock indices to analyze (None = all)

    Returns:
        List of ComponentContribution for each analyzed stock
    """
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(graph_data, enable_seat_branch=True)

    # Get all values as numpy arrays
    base_rank = outputs["base_rank_score"].cpu().numpy()
    seat_delta = outputs["seat_rank_delta"].cpu().numpy()
    gate = outputs["seat_gate"].cpu().numpy()
    final_rank = outputs["final_rank_score"].cpu().numpy()

    base_return = outputs["base_return_prediction"].cpu().numpy()
    seat_ret_delta = outputs["seat_return_delta"].cpu().numpy()
    final_return = outputs["final_return_prediction"].cpu().numpy()

    # Default to all stocks
    if target_indices is None:
        target_indices = list(range(len(base_rank)))

    contributions = []
    for idx in target_indices:
        gate_val = gate[idx]
        seat_contrib = gate_val * seat_delta[idx]

        # Seat contribution ratio: how much seat modified base
        seat_contrib_ratio = (
            abs(seat_contrib) / (abs(base_rank[idx]) + 1e-8)
            if base_rank[idx] != 0
            else 0.0
        )

        contributions.append(
            ComponentContribution(
                base_rank_score=float(base_rank[idx]),
                seat_rank_delta=float(seat_delta[idx]),
                seat_gate_value=float(gate_val),
                final_rank_score=float(final_rank[idx]),
                seat_contribution_ratio=float(seat_contrib_ratio),
                gate_activation_ratio=float(gate_val),
                base_return=float(base_return[idx]),
                seat_return_delta=float(seat_ret_delta[idx]),
                final_return=float(final_return[idx]),
            )
        )

    return contributions


def analyze_gate_activations(
    model: DTSeatHGTv3Model,
    graph_data: Any,
) -> GateAnalysis:
    """Analyze seat gate activation patterns.

    Args:
        model: Trained DT-SeatHGT v3 model
        graph_data: PyG HeteroData graph

    Returns:
        GateAnalysis with statistics and interpretation
    """
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(graph_data, enable_seat_branch=True)

    gate_values = outputs["seat_gate"].cpu().numpy()
    gate_list = gate_values.tolist()

    gate_mean = float(gate_values.mean())
    gate_std = float(gate_values.std())
    gate_min = float(gate_values.min())
    gate_max = float(gate_values.max())

    high_count = int((gate_values > 0.5).sum())
    low_count = int((gate_values <= 0.5).sum())

    # Generate interpretation
    if gate_mean > 0.7:
        interpretation = "High gate activation: seat branch strongly influences predictions"
    elif gate_mean > 0.4:
        interpretation = "Moderate gate activation: base and seat both contribute"
    elif gate_mean > 0.2:
        interpretation = "Low gate activation: base branch dominates, seat provides refinement"
    else:
        interpretation = "Minimal gate activation: base branch primarily drives predictions"

    return GateAnalysis(
        gate_values=gate_list,
        gate_mean=gate_mean,
        gate_std=gate_std,
        gate_min=gate_min,
        gate_max=gate_max,
        high_activation_count=high_count,
        low_activation_count=low_count,
        interpretation=interpretation,
    )


def explain_prediction(
    model: DTSeatHGTv3Model,
    graph_data: Any,
    target_index: int,
) -> dict[str, Any]:
    """Generate comprehensive explanation for a single prediction.

    Args:
        model: Trained DT-SeatHGT v3 model
        graph_data: PyG HeteroData graph
        target_index: Stock index to explain

    Returns:
        Dictionary with comprehensive explanation
    """
    model.eval()

    # Collect contributions
    contributions = analyze_component_contributions(
        model, graph_data, target_indices=[target_index]
    )
    contrib = contributions[0]

    # Gate analysis for all stocks
    gate_analysis = analyze_gate_activations(model, graph_data)

    # Get raw features for context
    stock_x = graph_data["stock"].x[target_index].cpu().numpy()

    explanation = {
        "target_index": target_index,
        "prediction": {
            "final_rank_score": contrib.final_rank_score,
            "final_return_prediction": contrib.final_return,
            "direction_logit": float(
                model(graph_data, enable_seat_branch=True)["direction_logit"][target_index]
            ),
        },
        "component_breakdown": {
            "base_branch": {
                "rank_score": contrib.base_rank_score,
                "return_prediction": contrib.base_return,
                "description": "Stock GRU + static MLP encoding",
            },
            "seat_branch": {
                "rank_delta": contrib.seat_rank_delta,
                "return_delta": contrib.seat_return_delta,
                "gate_value": contrib.seat_gate_value,
                "effective_contribution": contrib.seat_rank_delta * contrib.seat_gate_value,
                "description": "Seat soft profile residual branch",
            },
            "seat_influence": {
                "contribution_ratio": contrib.seat_contribution_ratio,
                "interpretation": _interpret_contribution_ratio(contrib.seat_contribution_ratio),
            },
        },
        "gate_statistics": {
            "this_stock": contrib.seat_gate_value,
            "population_mean": gate_analysis.gate_mean,
            "population_std": gate_analysis.gate_std,
            "interpretation": gate_analysis.interpretation,
        },
        "feature_context": {
            "stock_feature_dim": len(stock_x),
            "note": "Use RelationMasker for edge-level attribution",
        },
    }

    return explanation


def _interpret_contribution_ratio(ratio: float) -> str:
    """Interpret seat contribution ratio."""
    if ratio < 0.05:
        return "Minimal seat influence - base dominates"
    elif ratio < 0.2:
        return "Low seat influence - base primary"
    elif ratio < 0.5:
        return "Moderate seat influence - balanced"
    elif ratio < 1.0:
        return "High seat influence - seat significant"
    else:
        return "Very high seat influence - seat dominates"


def generate_explainability_report(
    model: DTSeatHGTv3Model,
    graph_data: Any,
    sample_name: str = "sample",
) -> str:
    """Generate human-readable explainability report.

    Args:
        model: Trained DT-SeatHGT v3 model
        graph_data: PyG HeteroData graph
        sample_name: Name for the sample being explained

    Returns:
        Formatted text report
    """
    contributions = analyze_component_contributions(model, graph_data)
    gate_analysis = analyze_gate_activations(model, graph_data)

    lines = [
        f"=== DT-SeatHGT v3 Explainability Report: {sample_name} ===",
        "",
        "## Gate Activation Analysis",
        f"  Mean: {gate_analysis.gate_mean:.4f}",
        f"  Std:  {gate_analysis.gate_std:.4f}",
        f"  Min:  {gate_analysis.gate_min:.4f}",
        f"  Max:  {gate_analysis.gate_max:.4f}",
        f"  High (>0.5): {gate_analysis.high_activation_count} stocks",
        f"  Low  (<=0.5): {gate_analysis.low_activation_count} stocks",
        f"  Interpretation: {gate_analysis.interpretation}",
        "",
        "## Component Contributions (first 5 stocks)",
        (
            f"  {'Idx':<4} {'Base Rank':<12} {'Seat Delta':<12} "
            f"{'Gate':<8} {'Final Rank':<12} {'Seat Ratio':<10}"
        ),
        "-" * 60,
    ]

    for i, c in enumerate(contributions[:5]):
        lines.append(
            f"  {i:<4} {c.base_rank_score:<12.4f} {c.seat_rank_delta:<12.4f} "
            f"{c.seat_gate_value:<8.4f} {c.final_rank_score:<12.4f} "
            f"{c.seat_contribution_ratio:<10.4f}"
        )

    lines.extend(["", "## Summary", ""])

    avg_seat_ratio = sum(c.seat_contribution_ratio for c in contributions) / len(contributions)
    if avg_seat_ratio < 0.1:
        lines.append("  -> Base branch dominates predictions")
    elif avg_seat_ratio < 0.3:
        lines.append("  -> Balanced base and seat contributions")
    else:
        lines.append("  -> Seat branch has significant influence")

    lines.extend(["", "=" * 50, ""])
    lines.append(
        "NOTE: For edge-level attribution, use RelationMasker "
        "from relation_contribution module."
    )

    return "\n".join(lines)


__all__ = [
    # New modules
    "RelationMasker",
    "EdgeMaskingResult",
    "SeatLevelMaskingResult",
    "RelationMaskingResult",
    "CriticalSubgraphResult",
    "CriticalSubgraphAnalyzer",
    "SeatInfluenceCluster",
    "explain_prediction_edge_contributions",
    "summarize_critical_subgraph",
    # Legacy components
    "ComponentContribution",
    "GateAnalysis",
    "analyze_component_contributions",
    "analyze_gate_activations",
    "explain_prediction",
    "generate_explainability_report",
]
