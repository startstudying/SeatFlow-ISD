"""Persistent-universe Top-K strategy with Rule B cost coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

PORTFOLIO_SIZE = 5
MAXIMUM_VOLUNTARY_REPLACEMENTS = 1
SWITCH_MARGIN = 0.0
COST_MULTIPLIER = 1.0
FORCED_EXIT_REASONS = frozenset(
    {"delisted", "permanently_untradable", "risk_rule", "critical_data_missing"}
)


@dataclass(frozen=True)
class ScoredStock:
    stock_id: str
    expected_return: float
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    slippage: float = 0.0


@dataclass(frozen=True)
class TradeDecision:
    trade_date: str
    previous_holdings: tuple[str, ...]
    persistent_candidates: tuple[str, ...]
    holding_scores: dict[str, float]
    new_candidate_scores: dict[str, float]
    proposed_replacements: tuple[tuple[str, str], ...]
    executed_replacements: tuple[tuple[str, str], ...]
    forced_exits: tuple[str, ...]
    decision_reason: str
    estimated_switch_cost: float
    holdings: tuple[str, ...]
    turnover: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_persistent_topk_rule_b(
    *,
    trade_date: str,
    previous_holdings: Sequence[str],
    candidates: Sequence[ScoredStock],
    forced_exit_reasons: Mapping[str, str] | None = None,
    portfolio_size: int = PORTFOLIO_SIZE,
    maximum_voluntary_replacements: int = MAXIMUM_VOLUNTARY_REPLACEMENTS,
    switch_margin: float = SWITCH_MARGIN,
    cost_multiplier: float = COST_MULTIPLIER,
) -> TradeDecision:
    """Select a fixed-size portfolio; candidate exit alone never forces selling."""

    if portfolio_size != PORTFOLIO_SIZE:
        raise ValueError("R4 v1 fixes portfolio_size=5")
    if maximum_voluntary_replacements != MAXIMUM_VOLUNTARY_REPLACEMENTS:
        raise ValueError("R4 v1 fixes maximum_voluntary_replacements=1")
    if switch_margin != SWITCH_MARGIN:
        raise ValueError("R4 v1 fixes switch_margin=0")
    if cost_multiplier != COST_MULTIPLIER:
        raise ValueError("R4 v1 fixes cost_multiplier=1")

    scores = {candidate.stock_id: candidate for candidate in candidates}
    if len(scores) != len(candidates):
        raise ValueError("candidate scores must contain unique stock_id values")
    previous = tuple(dict.fromkeys(str(stock_id) for stock_id in previous_holdings))
    reason_by_stock = forced_exit_reasons or {}
    forced_exits = tuple(
        stock_id
        for stock_id in previous
        if reason_by_stock.get(stock_id) in FORCED_EXIT_REASONS
    )
    missing_holding_scores = sorted(
        stock_id
        for stock_id in previous
        if stock_id not in scores and stock_id not in forced_exits
    )
    if missing_holding_scores:
        raise ValueError(
            f"all retained holdings require current scores: {missing_holding_scores}"
        )
    retained = [stock_id for stock_id in previous if stock_id not in forced_exits]
    holding_scores = {
        stock_id: scores[stock_id].expected_return
        for stock_id in previous
        if stock_id in scores
    }
    new_candidate_scores = {
        stock_id: score.expected_return
        for stock_id, score in scores.items()
        if stock_id not in retained
    }

    proposed: list[tuple[str, str]] = []
    executed: list[tuple[str, str]] = []
    estimated_switch_cost = 0.0
    if retained:
        replacement = _best_cost_covered_replacement(
            retained=retained,
            scores=scores,
            switch_margin=switch_margin,
        )
        if replacement is not None:
            sell_stock, buy_stock, cost = replacement
            proposed.append((sell_stock, buy_stock))
            executed.append((sell_stock, buy_stock))
            retained.remove(sell_stock)
            retained.append(buy_stock)
            estimated_switch_cost += cost

    for _forced_stock in forced_exits:
        fill_stock = _best_available_fill(retained, scores)
        if fill_stock is not None:
            retained.append(fill_stock)

    final_holdings = tuple(_top_fill_to_size(retained, scores, portfolio_size))
    if len(final_holdings) != portfolio_size:
        raise ValueError("insufficient scoreable candidates to maintain portfolio_size=5")
    previous_set = set(previous)
    turnover = len(previous_set - set(final_holdings)) / portfolio_size
    reason = "hold"
    if forced_exits:
        reason = "forced_exit"
    if executed:
        reason = "cost_covered_replacement"
    return TradeDecision(
        trade_date=trade_date,
        previous_holdings=previous,
        persistent_candidates=tuple(sorted(scores)),
        holding_scores=holding_scores,
        new_candidate_scores=new_candidate_scores,
        proposed_replacements=tuple(proposed),
        executed_replacements=tuple(executed),
        forced_exits=forced_exits,
        decision_reason=reason,
        estimated_switch_cost=estimated_switch_cost,
        holdings=final_holdings,
        turnover=turnover,
    )


def _best_cost_covered_replacement(
    *,
    retained: Sequence[str],
    scores: Mapping[str, ScoredStock],
    switch_margin: float,
) -> tuple[str, str, float] | None:
    worst_holding = min(
        (scores[stock_id] for stock_id in retained if stock_id in scores),
        key=lambda score: score.expected_return,
        default=None,
    )
    if worst_holding is None:
        return None
    available = [score for stock_id, score in scores.items() if stock_id not in retained]
    if not available:
        return None
    best_new = max(available, key=lambda score: score.expected_return)
    cost = best_new.buy_cost + worst_holding.sell_cost + best_new.slippage
    if best_new.expected_return - worst_holding.expected_return > cost + switch_margin:
        return (worst_holding.stock_id, best_new.stock_id, cost)
    return None


def _best_available_fill(retained: Sequence[str], scores: Mapping[str, ScoredStock]) -> str | None:
    retained_set = set(retained)
    available = [score for stock_id, score in scores.items() if stock_id not in retained_set]
    if not available:
        return None
    return max(available, key=lambda score: score.expected_return).stock_id


def _top_fill_to_size(
    retained: Sequence[str],
    scores: Mapping[str, ScoredStock],
    portfolio_size: int,
) -> list[str]:
    result = list(dict.fromkeys(retained))
    for score in sorted(scores.values(), key=lambda item: item.expected_return, reverse=True):
        if len(result) >= portfolio_size:
            break
        if score.stock_id not in result:
            result.append(score.stock_id)
    return result[:portfolio_size]
