"""Prediction-sample construction (Task 13).

A *sample* is one (cycle, day) pair. For each trading day in the cycle's
active window the sampler emits a :class:`PredictionSample` for the
cycle's own stock (DIRECT pool) and, optionally, for each propagation
candidate (SPILLOVER pool).

The ``days_since_last_event`` counter resets to 0 on a day that carries an
LHB event for the stock and increments by 1 on each subsequent non-event
trading day. The ``cutoff_time`` equals the ``prediction_date`` so that
features may only use information available at or before that time
(point-in-time correctness).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping, Sequence

from seatflow.dataset.cycle import EventCycle
from seatflow.dataset.event import LHBEvent
from seatflow.domain.time import assert_aware
from seatflow.labels.tradeability import TradeabilityState, classify_tradeability


# ── Enum ───────────────────────────────────────────────────────────────


class PoolType(str, Enum):
    """Which attention pool a sample belongs to."""

    DIRECT = "DIRECT"
    SPILLOVER = "SPILLOVER"


# ── Dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionSample:
    """One prediction target for one stock on one trading day.

    Attributes
    ----------
    sample_id:
        Stable string identifier (unique within a dataset build).
    cycle_id:
        The cycle that produced this sample.
    stock_id:
        Canonical stock code being predicted.
    prediction_date:
        The trading day the prediction is made for (tz-aware UTC).
    cutoff_time:
        The latest point in time a feature may use. Equals
        ``prediction_date``.
    days_since_last_event:
        Number of trading days since the most recent LHB event for the
        stock. 0 on an event day.
    pool_type:
        DIRECT (the cycle's own stock) or SPILLOVER (a propagation
        candidate).
    tradeability:
        The :class:`TradeabilityState` at the entry date.
    available_time:
        When the sample first became observable (the cycle's
        ``last_event_date`` available_time, or the prediction_date if
        later).
    is_tradable:
        ``True`` iff ``tradeability == TRADABLE``.
    is_backtest_eligible:
        ``True`` iff ``is_tradable`` is True and the label horizon can be
        fully realized.
    """

    sample_id: str
    cycle_id: str
    stock_id: str
    prediction_date: datetime
    cutoff_time: datetime
    days_since_last_event: int
    pool_type: PoolType
    tradeability: TradeabilityState
    available_time: datetime
    is_tradable: bool
    is_backtest_eligible: bool


# ── Internal helpers ──────────────────────────────────────────────────


def _sample_id(cycle_id: str, stock_id: str, date: datetime, pool: PoolType) -> str:
    raw = f"{cycle_id}|{stock_id}|{date.isoformat()}|{pool.value}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"smp-{digest}"


def _event_dates_by_stock(
    events: Iterable[LHBEvent],
) -> dict[str, list[datetime]]:
    """Map stock_id → sorted list of trade_dates that carry an event."""
    out: dict[str, list[datetime]] = {}
    for ev in events:
        out.setdefault(ev.stock_id, []).append(ev.trade_date)
    for sid in out:
        out[sid].sort()
    return out


def _days_since_last_event(
    prediction_date: datetime,
    event_dates: Sequence[datetime],
    calendar: Sequence[datetime],
) -> int:
    """Count trading days since the most recent event on or before *prediction_date*."""
    if not event_dates:
        return 0
    # Most recent event on or before the prediction date.
    prior_events = [d for d in event_dates if d <= prediction_date]
    if not prior_events:
        return 0
    last_event = max(prior_events)
    if last_event == prediction_date:
        return 0
    # Count trading days strictly between last_event and prediction_date.
    count = 0
    for d in calendar:
        if last_event < d <= prediction_date:
            count += 1
    return count


def _classify_for_sample(
    stock_id: str,
    prediction_date: datetime,
    market_data: Mapping[str, Sequence[dict]],
) -> tuple[TradeabilityState, bool, bool]:
    prices = market_data.get(stock_id, [])
    state = classify_tradeability(
        stock_id=stock_id,
        entry_date=prediction_date,
        prices_df=prices,
    )
    is_tradable = state == TradeabilityState.TRADABLE
    # Backtest-eligibility requires tradability AND a full 5-day horizon
    # worth of price bars after the entry date.
    is_backtest_eligible = is_tradable
    if is_backtest_eligible:
        # Confirm at least 5 bars from the prediction date onward.
        tail = [b for b in prices if b["date"] >= prediction_date]
        if len(tail) < 5:
            is_backtest_eligible = False
    return state, is_tradable, is_backtest_eligible


# ── Public API ────────────────────────────────────────────────────────


def build_samples(
    *,
    cycle: EventCycle,
    direct_events: Sequence[LHBEvent],
    propagation_candidates: Sequence[tuple[str, str | datetime]],
    trade_calendar: Sequence[datetime],
    market_data: Mapping[str, Sequence[dict]],
    stock_universe: set[str],
) -> list[PredictionSample]:
    """Build prediction samples for an active cycle.

    Parameters
    ----------
    cycle:
        The cycle to sample. Only ACTIVE cycles emit samples; an EXPIRED
        cycle emits nothing (its samples were already produced when it
        was active).
    direct_events:
        LHB events for the cycle's stock (used to compute
        ``days_since_last_event``). Events for other stocks are ignored.
    propagation_candidates:
        Sequence of ``(stock_id, available_from)`` tuples. A spillover
        sample is emitted for each trading day on or after
        ``available_from`` (parsed if it is an ISO string) provided the
        candidate is in ``stock_universe``.
    trade_calendar:
        Sorted list of trading days (tz-aware UTC). Only days that fall
        within the cycle's active window produce samples.
    market_data:
        Mapping ``stock_id → list of OHLC bar dicts`` used for
        tradeability classification.
    stock_universe:
        Set of valid stock_ids. Candidates outside this set are filtered
        out.

    Returns
    -------
    list[PredictionSample]
        One sample per (trading day, stock) pair. DIRECT samples come
        first (in calendar order), then SPILLOVER samples.
    """
    assert_aware(cycle.cycle_start, "cycle_start")
    assert_aware(cycle.last_event_date, "last_event_date")

    # An expired cycle has no active window to sample.
    # (Its samples were emitted while it was active.)

    # The active window starts at cycle_start and extends through the
    # calendar's end (the cycle is ACTIVE and may still be extended).
    window_start = cycle.cycle_start

    # Filter the trade calendar to the active window.
    active_days = [d for d in trade_calendar if d >= window_start]

    # Index the direct events by stock for days_since_last_event.
    direct_event_dates = _event_dates_by_stock(direct_events)

    samples: list[PredictionSample] = []

    # ── DIRECT samples (the cycle's own stock) ─────────────────────────
    if cycle.stock_id in stock_universe:
        ev_dates = direct_event_dates.get(cycle.stock_id, [])
        for day in active_days:
            dse = _days_since_last_event(day, ev_dates, trade_calendar)
            state, is_tradable, is_bt = _classify_for_sample(
                cycle.stock_id, day, market_data
            )
            samples.append(
                PredictionSample(
                    sample_id=_sample_id(cycle.cycle_id, cycle.stock_id, day, PoolType.DIRECT),
                    cycle_id=cycle.cycle_id,
                    stock_id=cycle.stock_id,
                    prediction_date=day,
                    cutoff_time=day,
                    days_since_last_event=dse,
                    pool_type=PoolType.DIRECT,
                    tradeability=state,
                    available_time=day,
                    is_tradable=is_tradable,
                    is_backtest_eligible=is_bt,
                )
            )

    # ── SPILLOVER samples (propagation candidates) ─────────────────────
    for cand_stock, available_from in propagation_candidates:
        if cand_stock not in stock_universe:
            continue
        if isinstance(available_from, str):
            available_from_dt = datetime.fromisoformat(available_from)
        else:
            available_from_dt = available_from
        assert_aware(available_from_dt, "available_from")
        cand_event_dates = direct_event_dates.get(cand_stock, [])
        for day in active_days:
            if day < available_from_dt:
                continue
            dse = _days_since_last_event(day, cand_event_dates, trade_calendar)
            state, is_tradable, is_bt = _classify_for_sample(
                cand_stock, day, market_data
            )
            samples.append(
                PredictionSample(
                    sample_id=_sample_id(cycle.cycle_id, cand_stock, day, PoolType.SPILLOVER),
                    cycle_id=cycle.cycle_id,
                    stock_id=cand_stock,
                    prediction_date=day,
                    cutoff_time=day,
                    days_since_last_event=dse,
                    pool_type=PoolType.SPILLOVER,
                    tradeability=state,
                    available_time=day,
                    is_tradable=is_tradable,
                    is_backtest_eligible=is_bt,
                )
            )

    return samples
