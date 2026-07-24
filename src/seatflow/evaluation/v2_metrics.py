"""Fail-closed metrics for executable three-class labels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, log2

from seatflow.datasets.pyg import (
    EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
    TensorizedWindowSample,
)
from seatflow.research import _spearman


@dataclass(frozen=True)
class V2EvaluationRow:
    as_of_date: str
    label: int
    excess_return: float
    probabilities: tuple[float, float, float]
    label_schema_version: str = EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA

    def __post_init__(self) -> None:
        if self.label_schema_version != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
            raise ValueError("V2 evaluation requires executable_excess_return_v2 labels")
        if self.label not in {-1, 0, 1}:
            raise ValueError("V2 labels must be -1, 0, or 1")
        if len(self.probabilities) != 3 or any(
            value < 0 or value > 1 for value in self.probabilities
        ):
            raise ValueError("V2 probabilities must contain three values in [0, 1]")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError("V2 probabilities must sum to one")


def evaluate_v2_rows(
    rows: Sequence[V2EvaluationRow], top_k: int = 5
) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("V2 evaluation requires at least one row")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    accuracy = sum(_prediction(row) == row.label for row in rows) / len(rows)
    by_date: dict[str, list[V2EvaluationRow]] = defaultdict(list)
    for row in rows:
        by_date[row.as_of_date].append(row)
    daily_ic = [
        _spearman([_positive(row) for row in group], [row.excess_return for row in group])
        for group in by_date.values()
    ]
    valid_ic = [value for value in daily_ic if value is not None]
    ranked = sorted(rows, key=_positive, reverse=True)[:top_k]
    precision = sum(row.label == 1 for row in ranked) / len(ranked)
    dcg = sum((2 ** (row.label + 1) - 1) / log2(index + 2) for index, row in enumerate(ranked))
    ideal = sorted(rows, key=lambda row: row.label, reverse=True)[:top_k]
    ideal_dcg = sum((2 ** (row.label + 1) - 1) / log2(index + 2) for index, row in enumerate(ideal))
    return {
        "sample_count": len(rows),
        "daily_rank_ic_mean": sum(valid_ic) / len(valid_ic) if valid_ic else None,
        "accuracy": accuracy,
        "precision_at_k": precision,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "top_k_mean_excess_return": sum(row.excess_return for row in ranked) / len(ranked),
    }


def v2_rows_from_prediction_snapshot(
    samples: Sequence[TensorizedWindowSample],
    snapshot: dict[str, object],
    *,
    horizon: int,
) -> tuple[V2EvaluationRow, ...]:
    """Convert only schema-valid three-class snapshots into V2 metric rows."""

    rows: list[V2EvaluationRow] = []
    for sample in samples:
        if sample.label_schema_version != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
            raise ValueError("V2 evaluation snapshot contains a non-V2 tensor sample")
        if sample.as_of_date is None:
            raise ValueError("V2 evaluation sample requires as_of_date")
        record = snapshot.get(sample.sample_id)
        if not isinstance(record, dict):
            raise ValueError("V2 prediction snapshot is missing a sample")
        directions = record.get("direction_logits")
        if not isinstance(directions, dict):
            raise ValueError("V2 prediction snapshot is missing direction logits")
        logits = directions.get(str(horizon))
        if not isinstance(logits, list) or len(logits) != 3:
            raise ValueError("V2 prediction snapshot requires exactly three direction logits")
        if horizon not in sample.labels or horizon not in sample.returns:
            raise ValueError("V2 evaluation sample is missing the requested horizon")
        rows.append(
            V2EvaluationRow(
                sample.as_of_date.isoformat(),
                sample.labels[horizon],
                sample.returns[horizon],
                _softmax((float(logits[0]), float(logits[1]), float(logits[2]))),
            )
        )
    return tuple(rows)


def _positive(row: V2EvaluationRow) -> float:
    return row.probabilities[2]


def _prediction(row: V2EvaluationRow) -> int:
    return (-1, 0, 1)[max(range(3), key=lambda index: row.probabilities[index])]


def _softmax(logits: tuple[float, float, float]) -> tuple[float, float, float]:
    maximum = max(logits)
    raw = tuple(exp(value - maximum) for value in logits)
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total
