"""Exposure audit helpers that evaluate coverage over target-stock feature schema only."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from seatflow.graph.tensorize import TensorizedGraph
from seatflow.graph.types import GraphNode, HeteroGraphSnapshot, NodeType


@dataclass(frozen=True)
class ExposureAuditResult:
    status: str
    inspected_nodes: int
    inspected_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    message: str


def _normalize_schema(feature_schema: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(str(name) for name in feature_schema or ())


def audit_stock_exposure(
    graph: HeteroGraphSnapshot,
    *,
    stock_feature_schema: Iterable[str] | None = None,
) -> ExposureAuditResult:
    """
    Audit how many stock-node features are present in the requested stock schema.

    Only stock nodes are inspected; non-stock nodes are ignored by construction.
    """
    allowed = set(_normalize_schema(stock_feature_schema))
    stock_nodes = tuple(node for node in graph.nodes if node.node_type == NodeType.STOCK)
    if not stock_nodes:
        return ExposureAuditResult(
            status="insufficient_evidence",
            inspected_nodes=0,
            inspected_features=tuple(sorted(allowed)),
            missing_features=tuple(sorted(allowed)),
            message="no stock nodes available for exposure audit",
        )
    if not allowed:
        return ExposureAuditResult(
            status="insufficient_evidence",
            inspected_nodes=len(stock_nodes),
            inspected_features=(),
            missing_features=(),
            message="stock feature schema is empty",
        )

    observed: set[str] = set()
    for node in stock_nodes:
        observed.update(
            feature for feature in node.features if feature in allowed and feature in node.features
        )
    missing = tuple(sorted(allowed.difference(observed)))
    if missing:
        return ExposureAuditResult(
            status="insufficient_evidence",
            inspected_nodes=len(stock_nodes),
            inspected_features=tuple(sorted(allowed)),
            missing_features=missing,
            message="stock node feature schema coverage is incomplete",
        )
    return ExposureAuditResult(
        status="confirmed",
        inspected_nodes=len(stock_nodes),
        inspected_features=tuple(sorted(allowed)),
        missing_features=(),
        message="stock feature schema coverage is sufficient",
    )


def audited_node_features(
    node: GraphNode, *, stock_only: bool = True
) -> dict[str, float | str | int | bool]:
    """
    Return a copy of auditable numeric/string features.

    The helper is intentionally permissive and does not mutate the original node.
    """
    if stock_only and node.node_type != NodeType.STOCK:
        return {}
    return dict(node.features)


def tensorized_target_exposures(
    graph: TensorizedGraph,
    target_node_index: int,
) -> dict[str, float]:
    """Extract only exposure fields physically present in the frozen tensor graph."""
    schema = graph.metadata.get("feature_schema")
    if not isinstance(schema, dict) or not isinstance(schema.get("stock"), list):
        raise ValueError("tensorized graph is missing the stock feature schema")
    stock_schema = [str(name) for name in schema["stock"]]
    values = graph.data["stock"].x[target_node_index]
    result = {
        name: float(values[index])
        for index, name in enumerate(stock_schema)
    }
    edge_count = 0
    edge_weight_sum = 0.0
    for edge_type in graph.data.edge_types:
        source, _, target = edge_type
        if {source, target} != {"seat", "stock"}:
            continue
        store = graph.data[edge_type]
        stock_indexes = store.edge_index[1] if target == "stock" else store.edge_index[0]
        mask = stock_indexes == target_node_index
        edge_count += int(mask.sum())
        if hasattr(store, "edge_weight"):
            edge_weight_sum += float(store.edge_weight[mask].sum())
    result["seat_edge_count"] = float(edge_count)
    result["seat_edge_weight_sum"] = edge_weight_sum
    return result


def compare_top5_exposures(
    rows: list[dict[str, object]],
    *,
    exposure_fields: Iterable[str],
    selected_field: str = "selected",
    return_field: str = "realized_return",
    fold_field: str = "fold",
) -> dict[str, object]:
    """Compare selected Top-5 rows with the full candidate pool."""
    fields = tuple(exposure_fields)
    available = tuple(
        field for field in fields if rows and all(field in row for row in rows)
    )
    missing = tuple(field for field in fields if field not in available)
    selected = [row for row in rows if bool(row.get(selected_field))]
    by_field: dict[str, object] = {}
    for field in available:
        selected_mean = _mapping_mean(selected, field)
        candidate_mean = _mapping_mean(rows, field)
        by_fold: dict[str, float] = {}
        for fold in sorted({str(row.get(fold_field)) for row in rows}):
            fold_rows = [row for row in rows if str(row.get(fold_field)) == fold]
            fold_selected = [row for row in fold_rows if bool(row.get(selected_field))]
            by_fold[fold] = _mapping_mean(fold_selected, field) - _mapping_mean(
                fold_rows, field
            )
        by_field[field] = {
            "top5_mean": selected_mean,
            "candidate_mean": candidate_mean,
            "difference": selected_mean - candidate_mean,
            "return_correlation": _correlation(rows, field, return_field),
            "fold_differences": by_fold,
        }
    return {
        "status": "supported" if available else "insufficient_evidence",
        "available_fields": list(available),
        "missing_fields": list(missing),
        "exposures": by_field,
    }


def _mapping_mean(rows: list[dict[str, object]], field: str) -> float:
    return fmean(float(str(row[field])) for row in rows) if rows else 0.0


def _correlation(
    rows: list[dict[str, object]],
    left_field: str,
    right_field: str,
) -> float | None:
    if len(rows) < 2:
        return None
    left = [float(str(row[left_field])) for row in rows]
    right = [float(str(row[right_field])) for row in rows]
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right) ** 0.5
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


__all__ = [
    "ExposureAuditResult",
    "audit_stock_exposure",
    "audited_node_features",
    "compare_top5_exposures",
    "tensorized_target_exposures",
]
