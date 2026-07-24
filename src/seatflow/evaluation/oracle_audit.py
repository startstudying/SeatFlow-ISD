"""Deterministic oracle helpers for ranking diagnostics.

The helpers here are intentionally small and data-structure agnostic: callers
pass any row mappings with the expected fields, and results are derived via
purely deterministic, order-independent ranking over each prediction date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from seatflow.evaluation.cost_attribution import (
    CostAttributionSummary,
    DailyCostAttribution,
    attribute_ranked_portfolios,
    summarize_cost_attribution,
)

Row = Mapping[str, object]


def _as_float(value: object) -> float:
    return float(str(value))


def _as_str(value: object) -> str:
    return str(value)


def _group_rows_by_date(
    rows: Sequence[Row],
    *,
    date_field: str,
) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        group = _as_str(row[date_field])
        grouped.setdefault(group, []).append(row)
    return grouped


def _top_k_gross_spread(
    rows: Sequence[Row],
    *,
    date_field: str,
    stock_field: str,
    score_field: str,
    return_field: str,
    top_k: int,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not rows:
        return 0.0
    grouped = _group_rows_by_date(rows, date_field=date_field)
    daily: list[float] = []
    for day in sorted(grouped):
        day_rows = grouped[day]
        if not day_rows:
            continue
        ordered = sorted(
            day_rows,
            key=lambda row: (-_as_float(row[score_field]), _as_str(row[stock_field])),
        )
        selected = ordered[:top_k]
        remainder = ordered[top_k:]
        selected_mean = fmean(_as_float(row[return_field]) for row in selected)
        remainder_mean = (
            fmean(_as_float(row[return_field]) for row in remainder)
            if remainder
            else 0.0
        )
        daily.append(selected_mean - remainder_mean)
    if not daily:
        return 0.0
    return fmean(daily)


def _score_delta(
    rows: Sequence[Row],
    *,
    first_score_field: str,
    second_score_field: str,
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    deltas = [
        abs(_as_float(row[first_score_field]) - _as_float(row[second_score_field]))
        for row in rows
    ]
    if not deltas:
        return 0.0, 0.0
    return fmean(deltas), max(deltas)


@dataclass(frozen=True)
class OracleResult:
    """Normalized oracle outcome used by audits and tests."""

    gross_spread: float
    top_k: int


@dataclass(frozen=True)
class OracleAuditResult:
    daily: tuple[DailyCostAttribution, ...]
    summary: CostAttributionSummary


def audit_oracle(
    rows: Sequence[Row],
    *,
    ranking_field: str,
    return_field: str,
    top_k: int = 5,
    date_field: str = "prediction_date",
    stock_field: str = "stock_id",
) -> OracleAuditResult:
    """Use one frozen field for ranking and one for realized economic return."""
    daily = attribute_ranked_portfolios(
        rows,
        top_k=top_k,
        date_field=date_field,
        stock_field=stock_field,
        score_field=ranking_field,
        return_field=return_field,
    )
    return OracleAuditResult(daily=daily, summary=summarize_cost_attribution(daily))


def audit_score_directions(
    rows: Sequence[Row],
    *,
    score_field: str = "score",
    return_field: str = "realized_return",
    top_k: int = 5,
) -> dict[str, OracleAuditResult]:
    augmented = [
        dict(row) | {"reversed_score": -_as_float(row[score_field])} for row in rows
    ]
    return {
        "original": audit_oracle(
            augmented,
            ranking_field=score_field,
            return_field=return_field,
            top_k=top_k,
        ),
        "reversed": audit_oracle(
            augmented,
            ranking_field="reversed_score",
            return_field=return_field,
            top_k=top_k,
        ),
    }


def label_oracle(
    rows: Sequence[Row],
    *,
    top_k: int = 5,
    date_field: str = "as_of_date",
    stock_field: str = "stock_id",
    score_field: str = "score",
    label_return_field: str = "label_return",
) -> OracleResult:
    """Compute top-k gross spread using label return as the realized value."""

    return OracleResult(
        gross_spread=_top_k_gross_spread(
            rows,
            date_field=date_field,
            stock_field=stock_field,
            score_field=score_field,
            return_field=label_return_field,
            top_k=top_k,
        ),
        top_k=top_k,
    )


def trade_return_oracle(
    rows: Sequence[Row],
    *,
    top_k: int = 5,
    date_field: str = "as_of_date",
    stock_field: str = "stock_id",
    score_field: str = "score",
    trade_return_field: str = "trade_return",
) -> OracleResult:
    """Compute top-k gross spread using trade-return style residuals."""

    return OracleResult(
        gross_spread=_top_k_gross_spread(
            rows,
            date_field=date_field,
            stock_field=stock_field,
            score_field=score_field,
            return_field=trade_return_field,
            top_k=top_k,
        ),
        top_k=top_k,
    )


@dataclass(frozen=True)
class ScoreDirectionDiagnostic:
    """Diagnostic for two correlated score columns in ranking rows."""

    mean_abs_delta: float
    max_abs_delta: float
    row_count: int
    max_delta_below_tolerance: bool


def score_direction_consistency(
    rows: Sequence[Row],
    *,
    score_field: str = "score",
    direction_score_field: str = "direction_probability",
    tolerance: float = 1e-12,
) -> ScoreDirectionDiagnostic:
    """Compare a score column with a companion direction score column."""

    if not rows:
        return ScoreDirectionDiagnostic(
            mean_abs_delta=0.0,
            max_abs_delta=0.0,
            row_count=0,
            max_delta_below_tolerance=True,
        )
    mean_delta, max_delta = _score_delta(
        rows,
        first_score_field=score_field,
        second_score_field=direction_score_field,
    )
    return ScoreDirectionDiagnostic(
        mean_abs_delta=mean_delta,
        max_abs_delta=max_delta,
        row_count=len(rows),
        max_delta_below_tolerance=max_delta <= tolerance,
    )


@dataclass(frozen=True)
class OriginalScoreDiagnostic:
    """Diagnostic comparing original score representation against active score."""

    score_field_mean_delta: float
    max_abs_delta: float
    within_tolerance: bool


def original_score_consistency(
    rows: Sequence[Row],
    *,
    original_score_field: str = "score",
    active_score_field: str = "selection_score",
    tolerance: float = 1e-12,
) -> OriginalScoreDiagnostic:
    """Compare model outputs (`selection_score`) with an original score column."""

    if not rows:
        return OriginalScoreDiagnostic(
            score_field_mean_delta=0.0,
            max_abs_delta=0.0,
            within_tolerance=True,
        )
    mean_delta, max_delta = _score_delta(
        rows,
        first_score_field=original_score_field,
        second_score_field=active_score_field,
    )
    return OriginalScoreDiagnostic(
        score_field_mean_delta=mean_delta,
        max_abs_delta=max_delta,
        within_tolerance=max_delta <= tolerance,
    )
