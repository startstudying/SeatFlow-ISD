"""Persistent candidate-universe construction for R4."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

MAXIMUM_EVENT_AGE_DAYS = 5

CandidateReason = str


@dataclass(frozen=True)
class PersistentCandidate:
    trade_date: str
    stock_id: str
    reason: CandidateReason
    is_current_event_candidate: bool
    is_recent_event_candidate: bool
    is_existing_holding: bool
    is_watchlist_candidate: bool
    event_age_trading_days: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PersistentUniverseDay:
    trade_date: str
    candidates: tuple[PersistentCandidate, ...]
    persistent_candidate_pool_size: int
    new_event_candidate_count: int
    recent_event_candidate_count: int
    holding_only_candidate_count: int
    watchlist_only_candidate_count: int
    candidate_pool_temporal_jaccard: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


def build_persistent_candidate_universe(
    *,
    daily_events: Mapping[str, Collection[str]],
    previous_holdings: Mapping[str, Collection[str]],
    trading_calendar: Sequence[str],
    watchlist: Mapping[str, Collection[str]] | None = None,
    maximum_event_age_days: int = MAXIMUM_EVENT_AGE_DAYS,
) -> tuple[PersistentUniverseDay, ...]:
    """Build U_t from current events, recent trading-day events, holdings, and watchlist.

    ``daily_events`` keys and ``trading_calendar`` are ordered trading dates.  Event
    age is measured by calendar index distance, never natural-day distance.
    ``previous_holdings[date]`` is the actual portfolio held before scoring ``date``.
    ``watchlist[date]`` is the user-defined research watchlist active on ``date``;
    watchlist stocks are kept scoreable even when no event or holding covers them.
    """

    if maximum_event_age_days != MAXIMUM_EVENT_AGE_DAYS:
        raise ValueError("R4 v1 fixes maximum_event_age_days=5; do not search H")
    if len(set(trading_calendar)) != len(trading_calendar):
        raise ValueError("trading_calendar contains duplicate dates")

    watchlist_by_date = watchlist or {}
    days: list[PersistentUniverseDay] = []
    prior_pool: set[str] | None = None
    for index, trade_date in enumerate(trading_calendar):
        current_events = _stock_set(daily_events.get(trade_date, ()))
        holdings = _stock_set(previous_holdings.get(trade_date, ()))
        watchlist_stocks = _stock_set(watchlist_by_date.get(trade_date, ()))
        recent_by_stock: dict[str, int] = {}
        start = max(0, index - maximum_event_age_days)
        for event_index in range(start, index + 1):
            event_date = trading_calendar[event_index]
            age = index - event_index
            for stock_id in _stock_set(daily_events.get(event_date, ())):
                previous_age = recent_by_stock.get(stock_id)
                if previous_age is None or age < previous_age:
                    recent_by_stock[stock_id] = age

        all_stock_ids = sorted(set(recent_by_stock) | holdings | watchlist_stocks)
        candidates = tuple(
            _candidate(
                trade_date=trade_date,
                stock_id=stock_id,
                current_events=current_events,
                holdings=holdings,
                watchlist_stocks=watchlist_stocks,
                recent_by_stock=recent_by_stock,
            )
            for stock_id in all_stock_ids
        )
        pool = {candidate.stock_id for candidate in candidates}
        days.append(
            PersistentUniverseDay(
                trade_date=trade_date,
                candidates=candidates,
                persistent_candidate_pool_size=len(candidates),
                new_event_candidate_count=sum(
                    candidate.is_current_event_candidate for candidate in candidates
                ),
                recent_event_candidate_count=sum(
                    candidate.is_recent_event_candidate
                    and not candidate.is_current_event_candidate
                    for candidate in candidates
                ),
                holding_only_candidate_count=sum(
                    candidate.is_existing_holding
                    and not candidate.is_current_event_candidate
                    and not candidate.is_recent_event_candidate
                    for candidate in candidates
                ),
                watchlist_only_candidate_count=sum(
                    candidate.is_watchlist_candidate
                    and not candidate.is_current_event_candidate
                    and not candidate.is_recent_event_candidate
                    and not candidate.is_existing_holding
                    for candidate in candidates
                ),
                candidate_pool_temporal_jaccard=(
                    1.0 if prior_pool is None else _jaccard(prior_pool, pool)
                ),
            )
        )
        prior_pool = pool
    return tuple(days)


def summarize_persistent_universe(
    days: Sequence[PersistentUniverseDay],
) -> dict[str, float | int | None]:
    if not days:
        return {
            "day_count": 0,
            "mean_persistent_candidate_pool_size": None,
            "mean_new_event_candidate_count": None,
            "mean_recent_event_candidate_count": None,
            "mean_holding_only_candidate_count": None,
            "mean_watchlist_only_candidate_count": None,
            "candidate_pool_temporal_jaccard": None,
        }
    return {
        "day_count": len(days),
        "mean_persistent_candidate_pool_size": fmean(
            day.persistent_candidate_pool_size for day in days
        ),
        "mean_new_event_candidate_count": fmean(
            day.new_event_candidate_count for day in days
        ),
        "mean_recent_event_candidate_count": fmean(
            day.recent_event_candidate_count for day in days
        ),
        "mean_holding_only_candidate_count": fmean(
            day.holding_only_candidate_count for day in days
        ),
        "mean_watchlist_only_candidate_count": fmean(
            day.watchlist_only_candidate_count for day in days
        ),
        "candidate_pool_temporal_jaccard": fmean(
            day.candidate_pool_temporal_jaccard for day in days
        ),
    }


def _candidate(
    *,
    trade_date: str,
    stock_id: str,
    current_events: set[str],
    holdings: set[str],
    watchlist_stocks: set[str],
    recent_by_stock: Mapping[str, int],
) -> PersistentCandidate:
    is_current = stock_id in current_events
    event_age = recent_by_stock.get(stock_id)
    is_recent = event_age is not None
    is_holding = stock_id in holdings
    is_watchlist = stock_id in watchlist_stocks
    reason_count = sum(
        (is_current, is_recent and not is_current, is_holding, is_watchlist)
    )
    if reason_count > 1:
        reason = "multiple_reasons"
    elif is_current:
        reason = "current_event"
    elif is_recent:
        reason = "recent_event"
    elif is_holding:
        reason = "existing_holding"
    elif is_watchlist:
        reason = "watchlist"
    else:  # pragma: no cover - guarded by caller's union
        raise AssertionError("candidate has no inclusion reason")
    return PersistentCandidate(
        trade_date=trade_date,
        stock_id=stock_id,
        reason=reason,
        is_current_event_candidate=is_current,
        is_recent_event_candidate=is_recent,
        is_existing_holding=is_holding,
        is_watchlist_candidate=is_watchlist,
        event_age_trading_days=event_age,
    )


def _stock_set(stocks: Collection[str]) -> set[str]:
    if isinstance(stocks, (str, bytes)):
        raise TypeError("stock collection must not be a string")
    return {str(stock) for stock in stocks}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)

