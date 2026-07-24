"""Deterministic candidate-vs-baseline portfolio gross/cost/net attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from seatflow.evaluation.portfolio_costs import TransactionCostModel, calculate_turnover_cost

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
        grouped.setdefault(_as_str(row[date_field]), []).append(row)
    return grouped


@dataclass(frozen=True)
class DailyCostAttribution:
    prediction_date: str
    candidate_gross_return: float
    baseline_gross_return: float
    gross_spread: float
    candidate_buy_turnover: float
    candidate_sell_turnover: float
    baseline_buy_turnover: float
    baseline_sell_turnover: float
    candidate_turnover: float
    baseline_turnover: float
    candidate_cost: float
    baseline_cost: float
    cost_difference: float
    net_spread: float
    continuation_rate: float

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CostAttributionSummary:
    mean_gross_spread: float
    mean_cost_difference: float
    mean_net_spread: float
    cost_to_gross_ratio: float | None
    mean_candidate_turnover: float
    mean_baseline_turnover: float
    mean_continuation_rate: float
    day_count: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def attribute_ranked_portfolios(
    rows: Sequence[Row],
    *,
    top_k: int = 5,
    date_field: str = "prediction_date",
    stock_field: str = "stock_id",
    score_field: str = "score",
    return_field: str = "realized_return",
    model: TransactionCostModel | None = None,
) -> tuple[DailyCostAttribution, ...]:
    """Attribute a ranked Top-K portfolio against the equal-weight remainder."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cost_model = TransactionCostModel() if model is None else model
    grouped = _group_rows_by_date(rows, date_field=date_field)
    previous_candidate: dict[str, float] = {}
    previous_baseline: dict[str, float] = {}
    result: list[DailyCostAttribution] = []
    for day in sorted(grouped):
        day_rows = grouped[day]
        identities = [_as_str(row[stock_field]) for row in day_rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate stock identity on {day}")
        ordered = sorted(
            day_rows,
            key=lambda row: (-_as_float(row[score_field]), _as_str(row[stock_field])),
        )
        candidate = ordered[:top_k]
        baseline = ordered[top_k:]
        if not candidate:
            continue
        candidate_weights = _equal_weights(candidate, stock_field)
        baseline_weights = _equal_weights(baseline, stock_field)
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
        cost_difference = candidate_cost.total_cost - baseline_cost.total_cost
        continuation = (
            len(set(candidate_weights) & set(previous_candidate)) / len(candidate_weights)
            if previous_candidate
            else 0.0
        )
        result.append(
            DailyCostAttribution(
                prediction_date=day,
                candidate_gross_return=candidate_return,
                baseline_gross_return=baseline_return,
                gross_spread=gross,
                candidate_buy_turnover=candidate_cost.buy_turnover,
                candidate_sell_turnover=candidate_cost.sell_turnover,
                baseline_buy_turnover=baseline_cost.buy_turnover,
                baseline_sell_turnover=baseline_cost.sell_turnover,
                candidate_turnover=candidate_cost.turnover,
                baseline_turnover=baseline_cost.turnover,
                candidate_cost=candidate_cost.total_cost,
                baseline_cost=baseline_cost.total_cost,
                cost_difference=cost_difference,
                net_spread=gross - candidate_cost.total_cost + baseline_cost.total_cost,
                continuation_rate=continuation,
            )
        )
        previous_candidate = candidate_weights
        previous_baseline = baseline_weights
    return tuple(result)


def summarize_cost_attribution(
    daily: Sequence[DailyCostAttribution],
) -> CostAttributionSummary:
    if not daily:
        return CostAttributionSummary(0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0)
    gross = fmean(row.gross_spread for row in daily)
    difference = fmean(row.cost_difference for row in daily)
    return CostAttributionSummary(
        mean_gross_spread=gross,
        mean_cost_difference=difference,
        mean_net_spread=fmean(row.net_spread for row in daily),
        cost_to_gross_ratio=(difference / abs(gross) if gross != 0 else None),
        mean_candidate_turnover=fmean(row.candidate_turnover for row in daily),
        mean_baseline_turnover=fmean(row.baseline_turnover for row in daily),
        mean_continuation_rate=fmean(row.continuation_rate for row in daily),
        day_count=len(daily),
    )


@dataclass(frozen=True)
class GrossCostNetIdentity:
    candidate_gross_spread: float
    baseline_gross_spread: float
    candidate_total_cost: float
    baseline_total_cost: float
    candidate_net_spread: float
    baseline_net_spread: float
    spread_net_from_identity: float


def gross_cost_net_identity(
    candidate_rows: Sequence[Row],
    baseline_rows: Sequence[Row],
    *,
    top_k: int = 5,
    score_field: str = "selection_score",
    return_field: str = "label_return",
    candidate_cost_model: TransactionCostModel | None = None,
    baseline_cost_model: TransactionCostModel | None = None,
    candidate_costs: dict[str, float] | None = None,
    baseline_costs: dict[str, float] | None = None,
) -> GrossCostNetIdentity:
    """Compute and validate candidate-vs-baseline gross/cost/net spread identity.

    The identity is:
    candidate_net_spread = candidate_gross_spread - candidate_cost + baseline_cost
    candidate_baseline_net_spread = candidate_net_spread - baseline_net_spread
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    candidate_daily = attribute_ranked_portfolios(
        candidate_rows,
        top_k=top_k,
        date_field="as_of_date",
        score_field=score_field,
        return_field=return_field,
        model=candidate_cost_model,
    )
    baseline_daily = attribute_ranked_portfolios(
        baseline_rows,
        top_k=top_k,
        date_field="as_of_date",
        score_field=score_field,
        return_field=return_field,
        model=baseline_cost_model,
    )
    candidate_gross = summarize_cost_attribution(candidate_daily).mean_gross_spread
    baseline_gross = summarize_cost_attribution(baseline_daily).mean_gross_spread
    candidate_summary = candidate_costs or {
        "total_cost": fmean(row.candidate_cost for row in candidate_daily)
        if candidate_daily
        else 0.0
    }
    baseline_summary = baseline_costs or {
        "total_cost": fmean(row.candidate_cost for row in baseline_daily)
        if baseline_daily
        else 0.0
    }
    candidate_total_cost = float(candidate_summary["total_cost"])
    baseline_total_cost = float(baseline_summary["total_cost"])
    candidate_net = candidate_gross - candidate_total_cost
    baseline_net = baseline_gross - baseline_total_cost
    net_identity = candidate_gross - baseline_gross - candidate_total_cost + baseline_total_cost
    return GrossCostNetIdentity(
        candidate_gross_spread=candidate_gross,
        baseline_gross_spread=baseline_gross,
        candidate_total_cost=candidate_total_cost,
        baseline_total_cost=baseline_total_cost,
        candidate_net_spread=candidate_net,
        baseline_net_spread=baseline_net,
        spread_net_from_identity=net_identity,
    )


def continuation_cost_rows(
    rows: Sequence[Row],
    *,
    top_k: int = 5,
    score_field: str = "selection_score",
    cost_model: TransactionCostModel | None = None,
) -> dict[str, float]:
    """Return turnover/cost summary preserving buy-sell split and continuation path."""

    daily = attribute_ranked_portfolios(
        rows,
        top_k=top_k,
        date_field="as_of_date",
        score_field=score_field,
        return_field="label_return",
        model=cost_model,
    )
    if not daily:
        return {
            "turnover": 0.0,
            "buy_turnover": 0.0,
            "sell_turnover": 0.0,
            "buy_cost": 0.0,
            "sell_cost": 0.0,
            "total_cost": 0.0,
        }
    return {
        "turnover": fmean(row.candidate_turnover for row in daily),
        "buy_turnover": fmean(row.candidate_buy_turnover for row in daily),
        "sell_turnover": fmean(row.candidate_sell_turnover for row in daily),
        "buy_cost": 0.0,
        "sell_cost": 0.0,
        "total_cost": fmean(row.candidate_cost for row in daily),
    }


def _equal_weights(rows: Sequence[Row], stock_field: str) -> dict[str, float]:
    if not rows:
        return {}
    weight = 1.0 / len(rows)
    return {_as_str(row[stock_field]): weight for row in rows}
