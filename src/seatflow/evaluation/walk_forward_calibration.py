"""Leakage-safe walk-forward calibration for R3.6 cost-aware ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import cast


@dataclass(frozen=True)
class CalibratedScore:
    prediction_date: str
    stock_id: str
    raw_score: float
    expected_gross_return: float
    matured_training_rows: int


def ridge_walk_forward_expected_returns(
    rows: Sequence[Mapping[str, object]],
    *,
    alpha: float = 1.0,
) -> tuple[CalibratedScore, ...]:
    """Fit an expanding one-feature Ridge model using only matured labels.

    A row is eligible for the training set on prediction date ``t`` only when
    its ``label_end_date`` is strictly earlier than ``t``.  The intercept is
    unpenalized and the single score coefficient uses the closed-form Ridge
    solution.  With no usable history, the expected return is zero.
    """

    if alpha < 0 or not isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    normalized = [_normalize_row(row) for row in rows]
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in normalized:
        by_date.setdefault(str(row["prediction_date"]), []).append(row)

    result: list[CalibratedScore] = []
    history: list[dict[str, object]] = []
    for prediction_date in sorted(by_date):
        current_date = date.fromisoformat(prediction_date)
        matured = [
            row
            for row in history
            if date.fromisoformat(str(row["label_end_date"])) < current_date
        ]
        intercept, slope = _fit_ridge(matured, alpha)
        for row in sorted(
            by_date[prediction_date], key=lambda item: str(item["stock_id"])
        ):
            raw_score = _as_float(row["score"])
            result.append(
                CalibratedScore(
                    prediction_date=prediction_date,
                    stock_id=str(row["stock_id"]),
                    raw_score=raw_score,
                    expected_gross_return=intercept + slope * raw_score,
                    matured_training_rows=len(matured),
                )
            )
        history.extend(by_date[prediction_date])
    return tuple(result)


def _fit_ridge(
    rows: Sequence[Mapping[str, object]], alpha: float
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    scores = [_as_float(row["score"]) for row in rows]
    returns = [_as_float(row["realized_return"]) for row in rows]
    score_mean = sum(scores) / len(scores)
    return_mean = sum(returns) / len(returns)
    numerator = sum(
        (score - score_mean) * (realized - return_mean)
        for score, realized in zip(scores, returns, strict=True)
    )
    denominator = sum((score - score_mean) ** 2 for score in scores) + alpha
    slope = numerator / denominator if denominator > 0 else 0.0
    return return_mean - slope * score_mean, slope


def _normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    required = (
        "prediction_date",
        "label_end_date",
        "stock_id",
        "score",
        "realized_return",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"calibration row missing fields: {missing}")
    prediction_date = str(row["prediction_date"])
    label_end_date = str(row["label_end_date"])
    date.fromisoformat(prediction_date)
    date.fromisoformat(label_end_date)
    score = _as_float(row["score"])
    realized = _as_float(row["realized_return"])
    if not isfinite(score) or not isfinite(realized):
        raise ValueError("calibration score and realized return must be finite")
    return {
        "prediction_date": prediction_date,
        "label_end_date": label_end_date,
        "stock_id": str(row["stock_id"]),
        "score": score,
        "realized_return": realized,
    }


def _as_float(value: object) -> float:
    return float(cast("str | float | int", value))
