"""Tests for PredictionSample construction (Task 13).

A *sample* is one (cycle, day) pair. For an active cycle that runs from
``cycle_start`` to ``last_event_date`` plus the active extension window,
the sampler emits one :class:`PredictionSample` for every trading day in
that range where the stock could be predicted.

Two pools are supported:

- ``DIRECT``: the stock that the LHB event refers to.
- ``SPILLOVER``: a propagation candidate (a co-moving stock) that the
  model can predict *because of* the original event.

The ``days_since_last_event`` counter resets to 0 on each LHB event for
the stock and increments by 1 on subsequent non-event days.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence


from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.event import LHBEvent
from seatflow.dataset.samples import (
    PoolType,
    PredictionSample,
    build_samples,
)
from seatflow.labels.tradeability import TradeabilityState


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _ev(
    event_id: str,
    stock_id: str,
    trade_date: str,
) -> LHBEvent:
    td = _utc(trade_date)
    return LHBEvent(
        event_id=event_id,
        stock_id=stock_id,
        trade_date=td,
        available_time=td,
        reason_codes=["日涨幅偏离值达7%的证券"],
        total_buy=1.0e8,
        total_sell=5.0e7,
        net_amount=5.0e7,
        turnover_rate=12.5,
        event_confidence=1.0,
        source_lineage="tushare",
    )


def _cycle(
    cycle_id: str,
    stock_id: str,
    start: str,
    last: str,
    *,
    status: CycleStatus = CycleStatus.ACTIVE,
    event_count: int = 1,
) -> EventCycle:
    return EventCycle(
        cycle_id=cycle_id,
        stock_id=stock_id,
        cycle_start=_utc(start),
        last_event_date=_utc(last),
        cycle_end=_utc(last) if status == CycleStatus.EXPIRED else None,
        event_count=event_count,
        status=status,
    )


def _bar(date: str, open_: float, close: float) -> dict:
    return {
        "date": _utc(date),
        "open": float(open_),
        "close": float(close),
        "high": float(max(open_, close)),
        "low": float(min(open_, close)),
        "volume": 1.0e6,
        "suspended": False,
    }


def _five_day_window(start: str) -> list[dict]:
    """Five consecutive trading days starting at *start* (UTC midnight)."""
    base = _utc(start)
    return [
        _bar((base + timedelta(days=i)).isoformat(), open_=10 + i, close=10.5 + i)
        for i in range(5)
    ]


def _trade_calendar(days: Sequence[str]) -> list[datetime]:
    return [_utc(d) for d in days]


def _build(
    *,
    cycle: EventCycle,
    direct_events: list[LHBEvent] | None = None,
    propagation_candidates: list[tuple[str, str]] | None = None,
    calendar: list[datetime] | None = None,
    market_data: dict | None = None,
    stock_universe: set[str] | None = None,
) -> list[PredictionSample]:
    """Build samples with sensible defaults."""
    if direct_events is None:
        direct_events = [_ev("e1", cycle.stock_id, cycle.cycle_start.isoformat())]
    if calendar is None:
        # Use a calendar covering 10 trading days around the cycle.
        base = cycle.cycle_start
        calendar = [base + timedelta(days=i) for i in range(10)]
    if market_data is None:
        market_data = {
            cycle.stock_id: _five_day_window(cycle.cycle_start.isoformat()),
            "000002.SZ": _five_day_window(cycle.cycle_start.isoformat()),
        }
    if stock_universe is None:
        stock_universe = {cycle.stock_id, "000002.SZ"}
    return build_samples(
        cycle=cycle,
        direct_events=direct_events,
        propagation_candidates=propagation_candidates or [],
        trade_calendar=calendar,
        market_data=market_data,
        stock_universe=stock_universe,
    )


# ── Sample construction ────────────────────────────────────────────────


class TestBuildSamples:
    def test_one_sample_per_trading_day_in_active_cycle(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        samples = _build(cycle=cycle)
        # Each event impulse is exactly D0..D0+4 exchange trading days.
        assert len(samples) == 5
        assert all(s.pool_type == PoolType.DIRECT for s in samples)
        assert all(s.cycle_id == "cyc-1" for s in samples)
        assert all(s.stock_id == "000001.SZ" for s in samples)

    def test_no_samples_for_non_trading_days(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Calendar contains only 3 trading days.
        calendar = _trade_calendar(
            [
                "2026-01-05T00:00:00+00:00",
                "2026-01-06T00:00:00+00:00",
                "2026-01-07T00:00:00+00:00",
            ]
        )
        samples = _build(cycle=cycle, calendar=calendar)
        assert len(samples) == 3
        assert [s.prediction_date for s in samples] == calendar

    def test_days_since_last_event_resets_on_event(self):
        """Day 0 = event day, day 1 = next, etc. Resets when a new event fires."""
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Second event for the same stock on day 3.
        direct_events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),
        ]
        calendar = [
            _utc("2026-01-05T00:00:00+00:00"),
            _utc("2026-01-06T00:00:00+00:00"),
            _utc("2026-01-07T00:00:00+00:00"),
            _utc("2026-01-08T00:00:00+00:00"),
            _utc("2026-01-09T00:00:00+00:00"),
        ]
        samples = _build(
            cycle=cycle,
            direct_events=direct_events,
            calendar=calendar,
        )
        # Each calendar day yields a direct sample for the cycle's stock.
        days_since = {s.prediction_date: s.days_since_last_event for s in samples}
        assert days_since[calendar[0]] == 0  # event day
        assert days_since[calendar[1]] == 1
        assert days_since[calendar[2]] == 2
        assert days_since[calendar[3]] == 0  # new event
        assert days_since[calendar[4]] == 1

    def test_samples_emit_for_propagation_candidates(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Spillover: 000002.SZ is influenced by the cycle on 000001.SZ.
        propagation_candidates = [("000002.SZ", "2026-01-05T00:00:00+00:00")]
        samples = _build(
            cycle=cycle,
            propagation_candidates=propagation_candidates,
        )
        # Five direct + five spillover targets.
        direct = [s for s in samples if s.pool_type == PoolType.DIRECT]
        spill = [s for s in samples if s.pool_type == PoolType.SPILLOVER]
        assert len(direct) == 5
        assert len(spill) == 5
        assert all(s.stock_id == "000002.SZ" for s in spill)
        assert all(s.cycle_id == "cyc-1" for s in spill)

    def test_tradeability_classified(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Provide 10 trading days of market data (matching the default
        # 10-day calendar) and force the entry bar to be suspended.
        base = _utc("2026-01-05T00:00:00+00:00")
        bars = [
            _bar((base + timedelta(days=i)).isoformat(), open_=10 + i, close=10.5 + i)
            for i in range(10)
        ]
        bars[0] = {
            "date": base,
            "open": 10, "close": 10, "high": 10, "low": 10,
            "volume": 0.0, "suspended": True,
        }
        market_data = {"000001.SZ": bars}
        samples = _build(cycle=cycle, market_data=market_data)
        # The first sample (entry day) is suspended.
        assert samples[0].tradeability == TradeabilityState.SUSPENDED_AT_ENTRY
        assert samples[0].is_tradable is False
        assert samples[0].is_backtest_eligible is False
        # Later samples with a full 5-day forward window are tradable.
        # (The final days lack a full horizon, so we only assert on days
        # 2..6 which each have >= 5 forward bars.)
        tradable_later = [
            s for s in samples
            if _utc("2026-01-06T00:00:00+00:00")
            <= s.prediction_date
            <= _utc("2026-01-10T00:00:00+00:00")
        ]
        assert all(s.is_tradable for s in tradable_later)

    def test_sample_ids_are_unique(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        samples = _build(
            cycle=cycle,
            propagation_candidates=[("000002.SZ", "2026-01-05T00:00:00+00:00")],
        )
        ids = [s.sample_id for s in samples]
        assert len(ids) == len(set(ids))

    def test_cutoff_time_equals_prediction_date(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        samples = _build(cycle=cycle)
        for s in samples:
            assert s.cutoff_time == s.prediction_date
            assert s.cutoff_time.tzinfo is not None

    def test_predicted_stock_outside_universe_filtered(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Spillover candidate not in the universe.
        propagation_candidates = [("999999.SZ", "2026-01-05T00:00:00+00:00")]
        samples = _build(
            cycle=cycle,
            propagation_candidates=propagation_candidates,
            stock_universe={"000001.SZ"},
        )
        # Only the five direct samples remain.
        assert all(s.stock_id == "000001.SZ" for s in samples)
        assert all(s.pool_type == PoolType.DIRECT for s in samples)

    def test_active_window_uses_five_exchange_days_across_holiday_gaps(self):
        cycle = _cycle(
            "cyc-1",
            "000001.SZ",
            "2026-01-30T00:00:00+00:00",
            "2026-02-20T00:00:00+00:00",
        )
        calendar = _trade_calendar(
            [
                "2026-01-30T00:00:00+00:00",
                "2026-02-02T00:00:00+00:00",
                "2026-02-03T00:00:00+00:00",
                "2026-02-09T00:00:00+00:00",
                "2026-02-10T00:00:00+00:00",
                "2026-02-11T00:00:00+00:00",
            ]
        )

        samples = _build(cycle=cycle, calendar=calendar)

        assert [sample.prediction_date for sample in samples] == calendar[:5]

    def test_cycle_extension_does_not_expand_original_impulse_window(self):
        calendar = _trade_calendar(
            [f"2026-01-{day:02d}T00:00:00+00:00" for day in range(5, 16)]
        )
        initial = _cycle(
            "cyc-1",
            "000001.SZ",
            "2026-01-05T00:00:00+00:00",
            "2026-01-05T00:00:00+00:00",
        )
        extended = _cycle(
            "cyc-1",
            "000001.SZ",
            "2026-01-05T00:00:00+00:00",
            "2026-01-12T00:00:00+00:00",
            event_count=2,
        )

        first = _build(cycle=initial, calendar=calendar)
        second = _build(cycle=extended, calendar=calendar)

        assert [sample.sample_id for sample in second] == [
            sample.sample_id for sample in first
        ]

    def test_support_only_stocks_never_become_prediction_samples(self):
        cycle = _cycle(
            "cyc-1",
            "000001.SZ",
            "2026-01-05T00:00:00+00:00",
            "2026-01-05T00:00:00+00:00",
        )

        samples = build_samples(
            cycle=cycle,
            direct_events=[_ev("e1", cycle.stock_id, cycle.cycle_start.isoformat())],
            propagation_candidates=[
                ("000002.SZ", "2026-01-05T00:00:00+00:00")
            ],
            trade_calendar=[cycle.cycle_start + timedelta(days=i) for i in range(10)],
            market_data={
                cycle.stock_id: _five_day_window(cycle.cycle_start.isoformat()),
                "000002.SZ": _five_day_window(cycle.cycle_start.isoformat()),
            },
            stock_universe={cycle.stock_id, "000002.SZ"},
            support_only_stock_ids={"000002.SZ"},
        )

        assert {sample.stock_id for sample in samples} == {"000001.SZ"}

    def test_empty_active_window_emits_no_samples(self):
        cycle = _cycle("cyc-1", "000001.SZ", "2026-01-05T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
        # Calendar contains no days within the active window.
        samples = _build(cycle=cycle, calendar=[])
        assert samples == []
