"""Deterministic Top-K and prediction-bucket economic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, log2
from statistics import fmean

from seatflow.evaluation.portfolio_costs import (
    TransactionCostModel,
    calculate_turnover_cost,
)

Row = Mapping[str, object]
TOP_K_SPECS: tuple[tuple[str, int | float], ...] = (
    ("top_1", 1),
    ("top_3", 3),
    ("top_5", 5),
    ("top_10", 10),
    ("top_20", 20),
    ("top_10pct", 0.10),
    ("top_20pct", 0.20),
)


def deterministic_rank(
    rows: Sequence[Row],
    *,
    score_field: str = "score",
    stock_field: str = "stock_id",
) -> list[Row]:
    return sorted(
        rows,
        key=lambda row: (-_as_float(row[score_field]), str(row[stock_field])),
    )


def partition_prediction_buckets(
    rows: Sequence[Row],
    *,
    bucket_count: int = 10,
    score_field: str = "score",
    stock_field: str = "stock_id",
) -> tuple[tuple[Row, ...], ...]:
    """Return disjoint, exhaustive best-to-worst score buckets."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    ordered = deterministic_rank(rows, score_field=score_field, stock_field=stock_field)
    buckets: list[list[Row]] = [[] for _ in range(bucket_count)]
    for index, row in enumerate(ordered):
        bucket = min(index * bucket_count // max(len(ordered), 1), bucket_count - 1)
        buckets[bucket].append(row)
    return tuple(tuple(bucket) for bucket in buckets)


def topk_diagnostics(
    rows: Sequence[Row],
    *,
    date_field: str = "prediction_date",
    stock_field: str = "stock_id",
    score_field: str = "score",
    return_field: str = "realized_return",
    model: TransactionCostModel | None = None,
) -> dict[str, object]:
    grouped = _group(rows, date_field)
    cost_model = TransactionCostModel() if model is None else model
    curves: dict[str, dict[str, object]] = {}
    daily_rows: list[dict[str, object]] = []
    for label, size in TOP_K_SPECS:
        previous_candidate: dict[str, float] = {}
        previous_baseline: dict[str, float] = {}
        values: list[dict[str, object]] = []
        for day in sorted(grouped):
            ordered = deterministic_rank(
                grouped[day], score_field=score_field, stock_field=stock_field
            )
            if not ordered:
                continue
            k = (
                min(int(size), len(ordered))
                if isinstance(size, int)
                else max(1, ceil(len(ordered) * size))
            )
            candidate, baseline = ordered[:k], ordered[k:]
            candidate_weights = _weights(candidate, stock_field)
            baseline_weights = _weights(baseline, stock_field)
            candidate_cost = calculate_turnover_cost(
                candidate_weights, previous_candidate, model=cost_model
            )
            baseline_cost = calculate_turnover_cost(
                baseline_weights, previous_baseline, model=cost_model
            )
            candidate_return = fmean(_as_float(row[return_field]) for row in candidate)
            baseline_return = (
                fmean(_as_float(row[return_field]) for row in baseline) if baseline else 0.0
            )
            gross = candidate_return - baseline_return
            row = {
                "prediction_date": day,
                "portfolio": label,
                "k": k,
                "candidate_gross_return": candidate_return,
                "baseline_gross_return": baseline_return,
                "gross_spread": gross,
                "candidate_cost": candidate_cost.total_cost,
                "baseline_cost": baseline_cost.total_cost,
                "net_spread": (
                    gross - candidate_cost.total_cost + baseline_cost.total_cost
                ),
            }
            values.append(row)
            daily_rows.append(row)
            previous_candidate = candidate_weights
            previous_baseline = baseline_weights
        curves[label] = {
            "mean_gross_spread": _mean(values, "gross_spread"),
            "mean_net_spread": _mean(values, "net_spread"),
            "day_count": len(values),
        }

    bucket_values: dict[int, list[float]] = {index: [] for index in range(1, 11)}
    ranking_metrics: list[dict[str, float]] = []
    for day in sorted(grouped):
        ordered = deterministic_rank(
            grouped[day], score_field=score_field, stock_field=stock_field
        )
        buckets = partition_prediction_buckets(
            ordered, score_field=score_field, stock_field=stock_field
        )
        for index, bucket in enumerate(buckets, start=1):
            if bucket:
                bucket_values[index].append(
                    fmean(_as_float(row[return_field]) for row in bucket)
                )
        ranking_metrics.append(
            _ranking_metrics(ordered, stock_field=stock_field, return_field=return_field)
        )
    return {
        "topk_curve": curves,
        "daily": daily_rows,
        "prediction_buckets": {
            str(index): {
                "mean_realized_return": (
                    fmean(values) if values else None
                ),
                "day_count": len(values),
            }
            for index, values in bucket_values.items()
        },
        "ranking_metrics": {
            name: fmean(row[name] for row in ranking_metrics) if ranking_metrics else 0.0
            for name in (
                "ndcg_at_5",
                "ndcg_at_10",
                "precision_at_5",
                "top5_realized_return_percentile",
                "rank_1_5_mean_return",
                "rank_6_10_mean_return",
            )
        },
    }


def _ranking_metrics(
    ordered: Sequence[Row],
    *,
    stock_field: str,
    return_field: str,
) -> dict[str, float]:
    returns = [_as_float(row[return_field]) for row in ordered]
    percentiles = _average_percentiles(returns)
    relevance = {
        str(row[stock_field]): percentile
        for row, percentile in zip(ordered, percentiles, strict=True)
    }
    ideal = sorted(relevance.values(), reverse=True)
    predicted = [relevance[str(row[stock_field])] for row in ordered]
    true_top5 = {
        stock
        for stock, _ in sorted(
            relevance.items(), key=lambda item: (-item[1], item[0])
        )[:5]
    }
    predicted_top5 = {str(row[stock_field]) for row in ordered[:5]}
    return {
        "ndcg_at_5": _ndcg(predicted, ideal, 5),
        "ndcg_at_10": _ndcg(predicted, ideal, 10),
        "precision_at_5": (
            len(true_top5 & predicted_top5) / len(predicted_top5)
            if predicted_top5
            else 0.0
        ),
        "top5_realized_return_percentile": (
            fmean(predicted[:5]) if predicted[:5] else 0.0
        ),
        "rank_1_5_mean_return": fmean(returns[:5]) if returns[:5] else 0.0,
        "rank_6_10_mean_return": fmean(returns[5:10]) if returns[5:10] else 0.0,
    }


def _average_percentiles(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    result = [0.0] * len(values)
    by_value: dict[float, list[int]] = {}
    for index, value in enumerate(values):
        by_value.setdefault(value, []).append(index)
    cursor = 0
    denominator = max(len(values) - 1, 1)
    for value in sorted(by_value):
        indexes = by_value[value]
        average_rank = (cursor + cursor + len(indexes) - 1) / 2
        percentile = average_rank / denominator if len(values) > 1 else 0.5
        for index in indexes:
            result[index] = percentile
        cursor += len(indexes)
    return result


def _ndcg(predicted: Sequence[float], ideal: Sequence[float], k: int) -> float:
    actual = sum(value / log2(index + 2) for index, value in enumerate(predicted[:k]))
    maximum = sum(value / log2(index + 2) for index, value in enumerate(ideal[:k]))
    return actual / maximum if maximum > 0 else 0.0


def _group(rows: Sequence[Row], date_field: str) -> dict[str, list[Row]]:
    result: dict[str, list[Row]] = {}
    for row in rows:
        result.setdefault(str(row[date_field]), []).append(row)
    return result


def _weights(rows: Sequence[Row], stock_field: str) -> dict[str, float]:
    if not rows:
        return {}
    weight = 1.0 / len(rows)
    return {str(row[stock_field]): weight for row in rows}


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return fmean(_as_float(row[field]) for row in rows) if rows else 0.0


def _as_float(value: object) -> float:
    return float(str(value))
