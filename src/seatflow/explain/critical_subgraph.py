"""Critical subgraph analysis for DT-SeatHGT v3.

This module identifies critical subgraphs by:
- Finding which seat→stock edges most influence predictions
- Identifying high-impact seat clusters
- Detecting edges with anomalous influence patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass
class CriticalSubgraphResult:
    """Result of critical subgraph identification."""

    # Top critical edges
    top_edges: list[dict[str, Any]] | None = None

    # Top critical seats
    top_seats: list[dict[str, Any]] | None = None

    # Edge type distribution
    edge_type_distribution: dict[str, int] | None = None

    # Aggregate statistics
    total_edges_analyzed: int = 0
    edges_with_significant_impact: int = 0

    # Impact thresholds
    significant_impact_threshold: float = 0.001


@dataclass
class SeatInfluenceCluster:
    """Cluster of seats with similar influence patterns."""

    cluster_id: int
    seat_ids: list[str]

    # Cluster statistics
    mean_impact: float = 0.0
    std_impact: float = 0.0
    total_stocks_affected: int = 0

    # Common characteristics
    common_direction: str | None = None  # "buys" or "sells" dominant


class CriticalSubgraphAnalyzer:
    """Analyze critical subgraphs in the seat-stock relation graph."""

    def __init__(self, model: nn.Module, significant_impact_threshold: float = 0.001) -> None:
        """Initialize analyzer.

        Args:
            model: DT-SeatHGT v3 model
            significant_impact_threshold: Minimum delta to consider significant
        """
        self.model = model
        self.model.eval()
        self.threshold = significant_impact_threshold

        # Import for masking
        from seatflow.explain.relation_contribution import RelationMasker
        self.masker = RelationMasker(model)

    def find_critical_edges(
        self,
        graph_data: Any,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Find the top K most critical seat→stock edges.

        Args:
            graph_data: HeteroData object
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID
            top_k: Number of top edges to return

        Returns:
            List of dicts with edge info and impact scores
        """
        edge_impact: list[dict[str, Any]] = []

        # Analyze all seat edges
        for edge_type in graph_data.edge_types:
            source_type, rel_name, target_type = edge_type

            if source_type != "seat":
                continue

            edge_store = graph_data[edge_type]
            if not hasattr(edge_store, "edge_index"):
                continue

            edge_index = edge_store.edge_index

            for edge_idx in range(edge_index.shape[1]):
                result = self.masker.analyze_single_edge(
                    graph_data, edge_type, edge_idx,
                    stock_id_map=stock_id_map,
                    seat_id_map=seat_id_map,
                )

                if abs(result.prediction_delta) >= self.threshold:
                    edge_impact.append({
                        "stock_id": result.stock_id,
                        "seat_id": result.seat_id,
                        "relation": result.relation,
                        "direction": result.direction,
                        "amount": result.amount,
                        "prediction_delta": result.prediction_delta,
                        "seat_gate": result.seat_gate,
                        "impact_score": abs(result.prediction_delta),
                    })

        # Sort by absolute impact
        edge_impact.sort(key=lambda x: x["impact_score"], reverse=True)

        return edge_impact[:top_k]

    def find_critical_seats(
        self,
        graph_data: Any,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the top K seats with highest aggregate impact.

        Args:
            graph_data: HeteroData object
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID
            top_k: Number of top seats to return

        Returns:
            List of dicts with seat info and impact scores
        """
        seat_impact: dict[int, dict[str, Any]] = {}

        # Analyze all seats
        for edge_type in graph_data.edge_types:
            source_type, rel_name, target_type = edge_type

            if source_type != "seat":
                continue

            edge_store = graph_data[edge_type]
            if not hasattr(edge_store, "edge_index"):
                continue

            edge_index = edge_store.edge_index

            for edge_idx in range(edge_index.shape[1]):
                seat_idx = edge_index[0, edge_idx].item()

                if seat_idx not in seat_impact:
                    seat_impact[seat_idx] = {
                        "seat_id": (
                            seat_id_map.get(seat_idx, str(seat_idx))
                            if seat_id_map
                            else str(seat_idx)
                        ),
                        "num_edges": 0,
                        "total_impact": 0.0,
                        "max_impact": 0.0,
                        "affected_stocks": set(),
                        "directions": {"buys": 0, "sells": 0},
                    }

                result = self.masker.analyze_single_edge(
                    graph_data, edge_type, edge_idx,
                    stock_id_map=stock_id_map,
                    seat_id_map=seat_id_map,
                )

                seat_impact[seat_idx]["num_edges"] += 1
                seat_impact[seat_idx]["total_impact"] += abs(result.prediction_delta)
                seat_impact[seat_idx]["max_impact"] = max(
                    seat_impact[seat_idx]["max_impact"],
                    abs(result.prediction_delta)
                )
                seat_impact[seat_idx]["affected_stocks"].add(result.stock_id)

                if result.direction:
                    seat_impact[seat_idx]["directions"][result.direction] += 1

        # Convert to list and sort
        seat_list = []
        for _seat_idx, data in seat_impact.items():
            dominant_direction = (
                "buys" if data["directions"]["buys"] > data["directions"]["sells"]
                else "sells" if data["directions"]["sells"] > data["directions"]["buys"]
                else "mixed"
            )
            seat_list.append({
                **data,
                "affected_stocks": list(data["affected_stocks"]),
                "mean_impact": (
                    data["total_impact"] / data["num_edges"]
                    if data["num_edges"] > 0
                    else 0.0
                ),
                "dominant_direction": dominant_direction,
            })

        seat_list.sort(key=lambda x: x["total_impact"], reverse=True)

        return seat_list[:top_k]

    def analyze_critical_subgraph(
        self,
        graph_data: Any,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
        top_k_edges: int = 20,
        top_k_seats: int = 10,
    ) -> CriticalSubgraphResult:
        """Analyze critical subgraphs in the seat-stock relation graph.

        Args:
            graph_data: HeteroData object
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID
            top_k_edges: Number of top edges to return
            top_k_seats: Number of top seats to return

        Returns:
            CriticalSubgraphResult with comprehensive analysis
        """
        top_edges = self.find_critical_edges(
            graph_data, stock_id_map, seat_id_map, top_k_edges
        )
        top_seats = self.find_critical_seats(
            graph_data, stock_id_map, seat_id_map, top_k_seats
        )

        # Count edge types
        edge_type_dist: dict[str, int] = {}
        for edge_type in graph_data.edge_types:
            source_type, rel_name, target_type = edge_type
            if source_type == "seat":
                edge_store = graph_data[edge_type]
                if hasattr(edge_store, "edge_index"):
                    key = f"{rel_name}"
                    edge_type_dist[key] = (
                        edge_type_dist.get(key, 0) + edge_store.edge_index.shape[1]
                    )

        # Count significant edges
        total_edges = sum(edge_type_dist.values())
        significant_edges = len(top_edges)

        return CriticalSubgraphResult(
            top_edges=top_edges,
            top_seats=top_seats,
            edge_type_distribution=edge_type_dist,
            total_edges_analyzed=total_edges,
            edges_with_significant_impact=significant_edges,
            significant_impact_threshold=self.threshold,
        )

    def identify_influence_clusters(
        self,
        graph_data: Any,
        stock_id_map: dict[int, str] | None = None,
        seat_id_map: dict[int, str] | None = None,
        num_clusters: int = 3,
    ) -> list[SeatInfluenceCluster]:
        """Identify clusters of seats with similar influence patterns.

        Args:
            graph_data: HeteroData object
            stock_id_map: Optional mapping from index to stock ID
            seat_id_map: Optional mapping from index to seat ID
            num_clusters: Number of clusters to identify

        Returns:
            List of SeatInfluenceCluster objects
        """
        # Get seat impact data
        seat_data = self.find_critical_seats(
            graph_data, stock_id_map, seat_id_map, top_k=100
        )

        if len(seat_data) < num_clusters:
            return []

        # Simple clustering based on mean_impact
        # In practice, you'd use K-means or hierarchical clustering
        impacts = [s["mean_impact"] for s in seat_data]
        impacts.sort(reverse=True)

        # Split into clusters
        cluster_size = len(impacts) // num_clusters
        clusters: list[list[dict[str, Any]]] = [[] for _ in range(num_clusters)]

        for seat in seat_data:
            impact = seat["mean_impact"]
            # Assign to highest cluster where impact exceeds threshold
            for i in range(num_clusters):
                threshold_idx = i * cluster_size
                if threshold_idx < len(impacts):
                    if impact >= impacts[threshold_idx]:
                        clusters[i].append(seat)
                        break

        # Build cluster objects
        result = []
        for i, cluster_seats in enumerate(clusters):
            if not cluster_seats:
                continue

            all_seat_ids = [s["seat_id"] for s in cluster_seats]
            impacts_in_cluster = [s["mean_impact"] for s in cluster_seats]

            # Count directions
            buys = sum(s["dominant_direction"] == "buys" for s in cluster_seats)
            sells = sum(s["dominant_direction"] == "sells" for s in cluster_seats)

            result.append(SeatInfluenceCluster(
                cluster_id=i,
                seat_ids=all_seat_ids,
                mean_impact=sum(impacts_in_cluster) / len(impacts_in_cluster),
                std_impact=(sum((x - sum(impacts_in_cluster)/len(impacts_in_cluster))**2
                           for x in impacts_in_cluster) / len(impacts_in_cluster)) ** 0.5,
                total_stocks_affected=sum(len(s["affected_stocks"]) for s in cluster_seats),
                common_direction="buys" if buys > sells else "sells" if sells > buys else "mixed",
            ))

        return result


def summarize_critical_subgraph(
    result: CriticalSubgraphResult,
) -> str:
    """Generate human-readable summary of critical subgraph analysis.

    Args:
        result: CriticalSubgraphResult from analyze_critical_subgraph

    Returns:
        Formatted text summary
    """
    lines = [
        "=== Critical Subgraph Analysis ===",
        "",
        f"Total edges analyzed: {result.total_edges_analyzed}",
        f"Edges with significant impact: {result.edges_with_significant_impact}",
        f"Impact threshold: {result.significant_impact_threshold}",
        "",
        "## Edge Type Distribution",
    ]

    for edge_type, count in (result.edge_type_distribution or {}).items():
        lines.append(f"  {edge_type}: {count}")

    lines.extend(["", "## Top Critical Edges (by impact)"])

    for i, edge in enumerate((result.top_edges or [])[:5]):
        lines.append(
            f"  {i+1}. Seat {edge['seat_id']} → Stock {edge['stock_id']} "
            f"(Δ={edge['prediction_delta']:.6f}, gate={edge['seat_gate']:.4f})"
        )

    lines.extend(["", "## Top Critical Seats (by aggregate impact)"])

    for i, seat in enumerate((result.top_seats or [])[:5]):
        lines.append(
            f"  {i+1}. Seat {seat['seat_id']}: "
            f"{seat['num_edges']} edges, "
            f"total impact={seat['total_impact']:.6f}, "
            f"max impact={seat['max_impact']:.6f}"
        )

    return "\n".join(lines)
