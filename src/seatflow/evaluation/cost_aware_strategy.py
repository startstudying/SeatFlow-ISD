"""Deterministic candidate-vs-baseline strategy simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from seatflow.evaluation.portfolio_costs import ranked_portfolio_cost_summary
from seatflow.evaluation.realistic_costs import RealisticCost, split_turnover_costs

Row = Mapping[str, object]


@dataclass(frozen=True)
class StrategyTransition:
    """Single-date output of a simulated path."""

    prediction_date: str
    path: str
    previous: tuple[str, ...]
    candidate: tuple[str, ...]
    reason: str
    switched: bool
    previous_gross: float
    candidate_gross: float
    cost: float
    net_utility: float


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _as_str(value: object) -> str:
    return str(value)


def _ranked_stocks(
    rows: Sequence[Row],
    *,
    stock_field: str,
    score_field: str,
) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (-_as_float(row[score_field]), _as_str(row[stock_field])),
    )
    return [_as_str(row[stock_field]) for row in ordered]


def simulate_p0(
    rows_by_date: Mapping[str, Sequence[Row]],
    *,
    stock_field: str = "stock_id",
    score_field: str = "score",
    top_k: int = 5,
) -> tuple[StrategyTransition, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    transitions: list[StrategyTransition] = []
    previous: tuple[str, ...] = ()
    for date in sorted(rows_by_date):
        ranked = _ranked_stocks(
            rows_by_date[date],
            stock_field=stock_field,
            score_field=score_field,
        )
        candidate = tuple(ranked[:top_k])
        candidate_gross = _sum_expected(
            rows_by_date[date],
            candidate=candidate,
            stock_field=stock_field,
            return_field=score_field,
        )
        transitions.append(
            StrategyTransition(
                prediction_date=date,
                path="p0",
                previous=previous,
                candidate=candidate,
                reason="top_k_recompute",
                switched=previous != candidate,
                previous_gross=_sum_expected(
                    rows_by_date[date],
                    candidate=previous,
                    stock_field=stock_field,
                    return_field=score_field,
                ),
                candidate_gross=candidate_gross,
                cost=0.0,
                net_utility=candidate_gross,
            )
        )
        previous = candidate
    return tuple(transitions)


def simulate_p1(
    rows_by_date: Mapping[str, Sequence[Row]],
    *,
    stock_field: str = "stock_id",
    score_field: str = "score",
    top_k: int = 5,
    candidate_exit_rule: str = "A",
) -> tuple[StrategyTransition, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidate_exit_rule != "A":
        raise ValueError("unsupported candidate_exit_rule")

    transitions: list[StrategyTransition] = []
    previous: tuple[str, ...] = ()
    for date in sorted(rows_by_date):
        ranked = _ranked_stocks(
            rows_by_date[date],
            stock_field=stock_field,
            score_field=score_field,
        )
        ranked_set = set(ranked)
        desired = tuple(ranked[:top_k])
        if not previous:
            candidate = desired
            reason = "first_day"
            cost_value = 0.0
        else:
            retained = [stock for stock in previous if stock in ranked_set]
            candidate_list = list(retained)
            # Rule A: replace any forced exits; this is always allowed and can be multiple.
            for stock in desired:
                if stock not in candidate_list:
                    if len(candidate_list) < top_k:
                        candidate_list.append(stock)
            candidate = tuple(dict.fromkeys(candidate_list[:top_k]))

            # Optional extra step: one voluntary replacement among retained names.
            rank_index = {stock: index for index, stock in enumerate(ranked)}
            extra_candidates = [stock for stock in desired if stock not in candidate]
            if len(candidate) == top_k and extra_candidates and previous:
                replaceable = [stock for stock in candidate if stock in previous]
                if replaceable:
                    replace_victim = max(
                        replaceable,
                        key=lambda stock: rank_index.get(stock, len(ranked)),
                    )
                    voluntary_in = extra_candidates[0]
                    if rank_index[voluntary_in] < rank_index[replace_victim]:
                        candidate = tuple(
                            voluntary_in if stock == replace_victim else stock
                            for stock in candidate
                        )
            if len(candidate) < top_k:
                for stock in desired:
                    if stock not in candidate:
                        candidate = (*candidate, stock)
                    if len(candidate) == top_k:
                        break

            cost_value = _portfolio_cost(
                previous=previous,
                candidate=candidate,
            )
            reason = "rule_a"

        candidate_gross = _sum_expected(
            rows_by_date[date],
            candidate=candidate,
            stock_field=stock_field,
            return_field=score_field,
        )
        if previous == candidate:
            cost_value = 0.0
        transitions.append(
            StrategyTransition(
                prediction_date=date,
                path="p1",
                previous=previous,
                candidate=candidate,
                reason=reason,
                switched=previous != candidate,
                previous_gross=_sum_expected(
                    rows_by_date[date],
                    candidate=previous,
                    stock_field=stock_field,
                    return_field=score_field,
                ),
                candidate_gross=candidate_gross,
                cost=cost_value,
                net_utility=candidate_gross - cost_value,
            )
        )
        previous = candidate
    return tuple(transitions)


def simulate_p2(
    rows_by_date: Mapping[str, Sequence[Row]],
    *,
    stock_field: str = "stock_id",
    score_field: str = "score",
    buy_rank: int = 5,
    hold_rank: int = 10,
) -> tuple[StrategyTransition, ...]:
    if buy_rank <= 0 or hold_rank <= 0:
        raise ValueError("buy_rank and hold_rank must be positive")
    if buy_rank > hold_rank:
        raise ValueError("buy_rank must be <= hold_rank")

    transitions: list[StrategyTransition] = []
    previous: tuple[str, ...] = ()
    for date in sorted(rows_by_date):
        ranked = _ranked_stocks(
            rows_by_date[date],
            stock_field=stock_field,
            score_field=score_field,
        )
        hold_set = set(ranked[:hold_rank])
        if not previous:
            candidate = tuple(ranked[:buy_rank])
        else:
            kept = tuple(stock for stock in previous if stock in hold_set)
            extras = [stock for stock in ranked[:hold_rank] if stock not in kept]
            candidate = tuple((kept + tuple(extras))[:buy_rank])

        candidate_gross = _sum_expected(
            rows_by_date[date],
            candidate=candidate,
            stock_field=stock_field,
            return_field=score_field,
        )
        cost_value = _portfolio_cost(
            previous=previous,
            candidate=candidate,
        )
        transitions.append(
            StrategyTransition(
                prediction_date=date,
                path="p2",
                previous=previous,
                candidate=candidate,
                reason="hold_rank_filter",
                switched=previous != candidate,
                previous_gross=_sum_expected(
                    rows_by_date[date],
                    candidate=previous,
                    stock_field=stock_field,
                    return_field=score_field,
                ),
                candidate_gross=candidate_gross,
                cost=cost_value,
                net_utility=candidate_gross - cost_value,
            )
        )
        previous = candidate
    return tuple(transitions)


def simulate_p3(
    rows_by_date: Mapping[str, Sequence[Row]],
    *,
    stock_field: str = "stock_id",
    score_field: str = "score",
    return_field: str = "expected_return",
    buy_rank: int = 5,
    hold_rank: int = 10,
    switch_margin: float = 0.0,
    cost: RealisticCost | None = None,
) -> tuple[StrategyTransition, ...]:
    if buy_rank <= 0 or hold_rank <= 0:
        raise ValueError("buy_rank and hold_rank must be positive")
    if buy_rank > hold_rank:
        raise ValueError("buy_rank must be <= hold_rank")

    transitions: list[StrategyTransition] = []
    previous: tuple[str, ...] = ()
    for date in sorted(rows_by_date):
        ranked = _ranked_stocks(
            rows_by_date[date],
            stock_field=stock_field,
            score_field=score_field,
        )
        hold = tuple(ranked[:hold_rank])
        buys = tuple(ranked[:buy_rank])
        if not previous:
            candidate = buys
            cost_value = 0.0
            reason = "first_day"
            net = _sum_expected(
                rows_by_date[date],
                candidate=candidate,
                stock_field=stock_field,
                return_field=return_field,
            )
        else:
            hold_allowed = tuple(stock for stock in previous if stock in hold)
            stay_candidate = tuple(
                hold_allowed + tuple(stock for stock in hold if stock not in hold_allowed)
            )[:buy_rank]
            stay_cost = _portfolio_cost(
                previous=previous,
                candidate=stay_candidate,
                cost=cost,
            )
            stay_gross = _sum_expected(
                rows_by_date[date],
                candidate=stay_candidate,
                stock_field=stock_field,
                return_field=return_field,
            )
            stay_net = stay_gross - stay_cost

            buy_candidate = buys
            buy_cost = _portfolio_cost(
                previous=previous,
                candidate=buy_candidate,
                cost=cost,
            )
            buy_gross = _sum_expected(
                rows_by_date[date],
                candidate=buy_candidate,
                stock_field=stock_field,
                return_field=return_field,
            )
            buy_net = buy_gross - buy_cost

            if buy_net > stay_net + switch_margin:
                candidate = buy_candidate
                cost_value = buy_cost
                net = buy_net
                reason = "cost_aware_switch"
            else:
                candidate = stay_candidate
                cost_value = stay_cost
                net = stay_net
                reason = "no_switch"
        transitions.append(
            StrategyTransition(
                prediction_date=date,
                path="p3",
                previous=previous,
                candidate=candidate,
                reason=reason,
                switched=previous != candidate,
                previous_gross=_sum_expected(
                    rows_by_date[date],
                    candidate=previous,
                    stock_field=stock_field,
                    return_field=return_field,
                ),
                candidate_gross=_sum_expected(
                    rows_by_date[date],
                    candidate=candidate,
                    stock_field=stock_field,
                    return_field=return_field,
                ),
                cost=cost_value,
                net_utility=net,
            )
        )
        previous = candidate
    return tuple(transitions)


def simulate_strategy_costs(
    transitions: Sequence[StrategyTransition],
    *,
    top_k: int,
) -> dict[str, float]:
    rows = [
        {
            "prediction_date": transition.prediction_date,
            "stock_id": stock_id,
            "score": 1.0 / max(top_k, 1),
        }
        for transition in transitions
        for stock_id in transition.candidate
    ]
    return ranked_portfolio_cost_summary(
        rows,
        top_k=top_k,
        date_field="prediction_date",
        stock_field="stock_id",
        score_field="score",
    )


def _sum_expected(
    rows: Sequence[Row],
    *,
    candidate: Sequence[str],
    stock_field: str,
    return_field: str,
) -> float:
    if not candidate:
        return 0.0
    row_map = _rows_to_map(rows, stock_field=stock_field)
    values = [
        _as_float(row_map[stock_id][return_field])
        for stock_id in candidate
        if stock_id in row_map
    ]
    return fmean(values) if values else 0.0


def _rows_to_map(rows: Sequence[Row], *, stock_field: str) -> dict[str, Row]:
    return {_as_str(row[stock_field]): row for row in rows}


def _portfolio_cost(
    *,
    previous: Sequence[str],
    candidate: Sequence[str],
    cost: RealisticCost | None = None,
) -> float:
    if not previous and not candidate:
        return 0.0
    previous_set = set(previous)
    candidate_set = set(candidate)
    weight_previous = 1.0 / len(previous) if previous else 0.0
    weight_candidate = 1.0 / len(candidate) if candidate else 0.0
    stock_set = previous_set | candidate_set
    buy_turnover = 0.0
    sell_turnover = 0.0
    for stock in stock_set:
        previous_weight = weight_previous if stock in previous_set else 0.0
        candidate_weight = weight_candidate if stock in candidate_set else 0.0
        delta = candidate_weight - previous_weight
        if delta > 0:
            buy_turnover += delta
        elif delta < 0:
            sell_turnover += -delta
    active_cost = RealisticCost() if cost is None else cost
    components = split_turnover_costs(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        cost=active_cost,
    )
    return components.total / active_cost.position_notional
