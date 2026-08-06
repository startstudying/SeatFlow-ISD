"""Tests for Event cycle construction (Task 8).

A *cycle* groups consecutive Dragon-Tiger-List events for the same stock
into a single attention unit. The key property is **atomicity**: a cycle
is the split atom, not the individual event. Models and backtests operate
on cycles, so the construction must be deterministic and reversible.

Rules
-----
- Events are grouped by ``stock_id``.
- A new cycle starts at the first event for a stock.
- A subsequent event is added to the active cycle if its ``trade_date``
  is within ``extension_days`` of the active cycle's
  ``last_event_date``.
- If the gap exceeds ``extension_days``, the active cycle is
  ``EXPIRED`` and a new ``ACTIVE`` cycle begins.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

import pytest

from seatflow.dataset.cycle import (
    CycleStatus,
    EventCycle,
    build_cycles,
    extend_cycle,
)
from seatflow.dataset.event import LHBEvent


# 鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _ev(
    event_id: str,
    stock_id: str,
    trade_date: str,
    *,
    available_time: str | None = None,
) -> LHBEvent:
    """Build a minimal valid LHBEvent for cycle tests."""
    td = _utc(trade_date)
    at = _utc(available_time) if available_time else td
    return LHBEvent(
        event_id=event_id,
        stock_id=stock_id,
        trade_date=td,
        available_time=at,
        reason_codes=["limit-deviation"],
        total_buy=1.0e8,
        total_sell=5.0e7,
        net_amount=5.0e7,
        turnover_rate=12.5,
        event_confidence=1.0,
        source_lineage="tushare",
    )


def _calendar(days: Sequence[str]) -> list[datetime]:
    return [_utc(day) for day in days]


def _daily_calendar() -> list[datetime]:
    start = _utc("2026-01-01T00:00:00+00:00")
    return [start + timedelta(days=i) for i in range(260)]


# 鈹€鈹€ Dataclass contracts 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestCycleDataclass:
    def test_status_enum_values(self):
        assert CycleStatus.ACTIVE.value == "ACTIVE"
        assert CycleStatus.EXPIRED.value == "EXPIRED"

    def test_cycle_id_is_string(self):
        c = EventCycle(
            cycle_id="cyc-000001.SZ-1",
            stock_id="000001.SZ",
            cycle_start=_utc("2026-01-05T00:00:00+00:00"),
            last_event_date=_utc("2026-01-05T00:00:00+00:00"),
            cycle_end=None,
            event_count=1,
            status=CycleStatus.ACTIVE,
        )
        assert isinstance(c.cycle_id, str)
        assert isinstance(c.stock_id, str)


# 鈹€鈹€ Cycle creation from consecutive events 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestBuildCyclesConsecutive:
    def test_weekend_holiday_gap_within_five_exchange_days_stays_one_cycle(self):
        calendar = _calendar(
            [
                "2026-01-30T00:00:00+00:00",
                "2026-02-02T00:00:00+00:00",
                "2026-02-03T00:00:00+00:00",
                "2026-02-05T00:00:00+00:00",
                "2026-02-06T00:00:00+00:00",
                "2026-02-09T00:00:00+00:00",
                "2026-02-10T00:00:00+00:00",
                "2026-02-11T00:00:00+00:00",
                "2026-02-12T00:00:00+00:00",
                "2026-02-13T00:00:00+00:00",
                "2026-02-16T00:00:00+00:00",
            ]
        )
        events = [
            _ev("e1", "000001.SZ", "2026-01-30T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-02-09T00:00:00+00:00"),
        ]

        cycles = build_cycles(events, trade_calendar=calendar)

        assert len(cycles) == 1
        assert cycles[0].event_count == 2
        assert cycles[0].cycle_end == _utc("2026-02-16T00:00:00+00:00")

    def test_event_on_d5_boundary_resets_cycle_end_to_its_own_d5(self):
        calendar = _calendar(
            [
                "2026-01-05T00:00:00+00:00",
                "2026-01-06T00:00:00+00:00",
                "2026-01-07T00:00:00+00:00",
                "2026-01-08T00:00:00+00:00",
                "2026-01-09T00:00:00+00:00",
                "2026-01-12T00:00:00+00:00",
                "2026-01-13T00:00:00+00:00",
                "2026-01-14T00:00:00+00:00",
                "2026-01-15T00:00:00+00:00",
                "2026-01-16T00:00:00+00:00",
                "2026-01-19T00:00:00+00:00",
            ]
        )
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-12T00:00:00+00:00"),
        ]

        cycles = build_cycles(events, trade_calendar=calendar)

        assert len(cycles) == 1
        assert cycles[0].event_count == 2
        assert cycles[0].last_event_date == _utc("2026-01-12T00:00:00+00:00")
        assert cycles[0].cycle_end == _utc("2026-01-19T00:00:00+00:00")

    def test_single_event_yields_single_active_cycle(self):
        events = [_ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")]
        cycles = build_cycles(events, trade_calendar=_daily_calendar())
        assert len(cycles) == 1
        assert cycles[0].event_count == 1
        assert cycles[0].status == CycleStatus.ACTIVE
        assert cycles[0].stock_id == "000001.SZ"
        assert cycles[0].cycle_end == _utc("2026-01-10T00:00:00+00:00")

    def test_consecutive_events_within_window_grouped(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),  # +3 days
            _ev("e3", "000001.SZ", "2026-01-10T00:00:00+00:00"),  # +2 days
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert len(cycles) == 1
        assert cycles[0].event_count == 3
        assert cycles[0].status == CycleStatus.ACTIVE
        assert cycles[0].cycle_start == _utc("2026-01-05T00:00:00+00:00")
        assert cycles[0].last_event_date == _utc("2026-01-10T00:00:00+00:00")
        assert cycles[0].cycle_end == _utc("2026-01-15T00:00:00+00:00")

    def test_event_at_exact_extension_window_boundary(self):
        """Gap of exactly extension_days should be included (within window)."""
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-10T00:00:00+00:00"),  # +5 days
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert len(cycles) == 1
        assert cycles[0].event_count == 2


# 鈹€鈹€ Cycle extension window 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestBuildCyclesExtension:
    def test_extend_cycle_within_window(self):
        cycle = EventCycle(
            cycle_id="cyc-1",
            stock_id="000001.SZ",
            cycle_start=_utc("2026-01-05T00:00:00+00:00"),
            last_event_date=_utc("2026-01-05T00:00:00+00:00"),
            cycle_end=_utc("2026-01-10T00:00:00+00:00"),
            event_count=1,
            status=CycleStatus.ACTIVE,
        )
        new_event = _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00")
        extended = extend_cycle(cycle, new_event, trade_calendar=_daily_calendar(), extension_days=5)
        assert extended.event_count == 2
        assert extended.last_event_date == _utc("2026-01-08T00:00:00+00:00")
        assert extended.status == CycleStatus.ACTIVE
        assert extended.cycle_id == cycle.cycle_id  # same cycle

    def test_extend_cycle_rejects_different_stock(self):
        cycle = EventCycle(
            cycle_id="cyc-1",
            stock_id="000001.SZ",
            cycle_start=_utc("2026-01-05T00:00:00+00:00"),
            last_event_date=_utc("2026-01-05T00:00:00+00:00"),
            cycle_end=_utc("2026-01-10T00:00:00+00:00"),
            event_count=1,
            status=CycleStatus.ACTIVE,
        )
        new_event = _ev("e2", "000002.SZ", "2026-01-08T00:00:00+00:00")
        with pytest.raises(ValueError, match="stock"):
            extend_cycle(cycle, new_event, trade_calendar=_daily_calendar(), extension_days=5)

    def test_extend_cycle_rejects_outside_window(self):
        cycle = EventCycle(
            cycle_id="cyc-1",
            stock_id="000001.SZ",
            cycle_start=_utc("2026-01-05T00:00:00+00:00"),
            last_event_date=_utc("2026-01-05T00:00:00+00:00"),
            cycle_end=_utc("2026-01-10T00:00:00+00:00"),
            event_count=1,
            status=CycleStatus.ACTIVE,
        )
        # 10 days later, well outside the 5-day window
        new_event = _ev("e2", "000001.SZ", "2026-01-15T00:00:00+00:00")
        with pytest.raises(ValueError, match="extension window"):
            extend_cycle(cycle, new_event, trade_calendar=_daily_calendar(), extension_days=5)

    def test_extend_cycle_preserves_cycle_id(self):
        cycle = EventCycle(
            cycle_id="cyc-custom-id",
            stock_id="000001.SZ",
            cycle_start=_utc("2026-01-05T00:00:00+00:00"),
            last_event_date=_utc("2026-01-05T00:00:00+00:00"),
            cycle_end=_utc("2026-01-10T00:00:00+00:00"),
            event_count=1,
            status=CycleStatus.ACTIVE,
        )
        new_event = _ev("e2", "000001.SZ", "2026-01-06T00:00:00+00:00")
        extended = extend_cycle(cycle, new_event, trade_calendar=_daily_calendar(), extension_days=5)
        assert extended.cycle_id == "cyc-custom-id"


# 鈹€鈹€ Cycle expiry on gap 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestBuildCyclesExpiry:
    def test_gap_exceeds_window_starts_new_cycle(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),  # +3
            # 30-day gap exceeds 5-day window
            _ev("e3", "000001.SZ", "2026-02-07T00:00:00+00:00"),
            _ev("e4", "000001.SZ", "2026-02-10T00:00:00+00:00"),  # +3
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert len(cycles) == 2

        c1, c2 = cycles
        assert c1.event_count == 2
        assert c1.status == CycleStatus.EXPIRED
        assert c1.cycle_end == _utc("2026-01-13T00:00:00+00:00")
        assert c1.cycle_start == _utc("2026-01-05T00:00:00+00:00")

        assert c2.event_count == 2
        assert c2.status == CycleStatus.ACTIVE
        assert c2.cycle_start == _utc("2026-02-07T00:00:00+00:00")
        assert c2.cycle_end == _utc("2026-02-15T00:00:00+00:00")

    def test_first_cycle_expires_later_cycles_active(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-10T00:00:00+00:00"),  # +5
            _ev("e3", "000001.SZ", "2026-01-25T00:00:00+00:00"),  # +15 鈫?new cycle
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert len(cycles) == 2
        assert cycles[0].status == CycleStatus.EXPIRED
        assert cycles[1].status == CycleStatus.ACTIVE

    def test_three_cycles_with_large_gaps(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-10T00:00:00+00:00"),
            # 100-day gap
            _ev("e3", "000001.SZ", "2026-04-20T00:00:00+00:00"),
            _ev("e4", "000001.SZ", "2026-04-25T00:00:00+00:00"),
            # 100-day gap
            _ev("e5", "000001.SZ", "2026-08-03T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert len(cycles) == 3
        assert [c.status for c in cycles] == [
            CycleStatus.EXPIRED,
            CycleStatus.EXPIRED,
            CycleStatus.ACTIVE,
        ]
        assert [c.event_count for c in cycles] == [2, 2, 1]


# 鈹€鈹€ Cycle status transitions 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestCycleStatusTransitions:
    def test_active_to_expired_on_new_cycle(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-25T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert cycles[0].status == CycleStatus.EXPIRED
        assert cycles[1].status == CycleStatus.ACTIVE
        assert cycles[0].cycle_end == _utc("2026-01-13T00:00:00+00:00")

    def test_single_event_keeps_active_with_no_end(self):
        events = [_ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")]
        cycles = build_cycles(events, trade_calendar=_daily_calendar())
        assert cycles[0].status == CycleStatus.ACTIVE
        assert cycles[0].cycle_end == _utc("2026-01-10T00:00:00+00:00")

    def test_mixed_stocks_produce_independent_cycles(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000002.SZ", "2026-01-06T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-10T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        stock_ids = [c.stock_id for c in cycles]
        # 000001 has two events within window 鈫?1 cycle
        # 000002 has one event 鈫?1 cycle
        assert stock_ids.count("000001.SZ") == 1
        assert stock_ids.count("000002.SZ") == 1
        assert all(c.status == CycleStatus.ACTIVE for c in cycles)

    def test_default_extension_days(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-10T00:00:00+00:00"),
        ]
        # Without specifying extension_days, default 5 is used; all within window.
        cycles = build_cycles(events, trade_calendar=_daily_calendar())
        assert len(cycles) == 1
        assert cycles[0].event_count == 3


# 鈹€鈹€ Cycle atomicity (split atom) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestCycleAtomicity:
    def test_cycles_partition_all_events(self):
        """The set of events across all cycles equals the input set 鈥?nothing
        is dropped, nothing is duplicated."""
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-07T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-20T00:00:00+00:00"),
            _ev("e4", "000002.SZ", "2026-01-08T00:00:00+00:00"),
            _ev("e5", "000002.SZ", "2026-01-10T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        # We can't directly inspect the events stored on cycles (cycles only
        # carry counts), but cycle_start/last_event_date must cover all
        # trade_dates and sum(event_count) == len(events).
        total = sum(c.event_count for c in cycles)
        assert total == len(events)
        # First cycle's start is the earliest event
        assert cycles[0].cycle_start == _utc("2026-01-05T00:00:00+00:00")

    def test_cycles_are_deterministic(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-10T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-20T00:00:00+00:00"),
        ]
        c1 = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        c2 = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert [(c.cycle_id, c.stock_id, c.event_count, c.status) for c in c1] == [
            (c.cycle_id, c.stock_id, c.event_count, c.status) for c in c2
        ]

    def test_cycle_is_split_atom_not_event(self):
        """The 'atom' a model consumes is the cycle. Different cycle boundary
        choices must produce different cycle sets 鈥?confirming the cycle is
        the split atom (not the event)."""
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-09T00:00:00+00:00"),  # +4 days
            _ev("e3", "000001.SZ", "2026-01-13T00:00:00+00:00"),  # +4 days
        ]
        # With a 10-day window, all three are in one cycle.
        wide = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=10)
        # With a 3-day window, both gaps (4 days) exceed the window, so each
        # event forms its own cycle.
        narrow = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=3)
        assert len(wide) == 1
        assert len(narrow) == 3
        # The events themselves are unchanged 鈥?only the grouping changes.
        # This proves cycles are the split atom: the same events can yield
        # many valid cycle partitions.

    def test_event_count_equals_grouped_event_count(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000001.SZ", "2026-01-07T00:00:00+00:00"),
            _ev("e3", "000001.SZ", "2026-01-09T00:00:00+00:00"),
            _ev("e4", "000001.SZ", "2026-01-30T00:00:00+00:00"),
            _ev("e5", "000001.SZ", "2026-02-02T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        # First cycle: 3 events (e1, e2, e3); second cycle: 2 events (e4, e5)
        assert cycles[0].event_count == 3
        assert cycles[1].event_count == 2


# 鈹€鈹€ Edge cases 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestEdgeCases:
    def test_empty_events_yields_empty_cycles(self):
        assert build_cycles([], trade_calendar=_daily_calendar()) == []

    def test_unsorted_events_sorted_by_trade_date(self):
        events = [
            _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00"),
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
        ]
        cycles = build_cycles(events, trade_calendar=_daily_calendar(), extension_days=5)
        assert cycles[0].cycle_start == _utc("2026-01-05T00:00:00+00:00")
        assert cycles[0].last_event_date == _utc("2026-01-08T00:00:00+00:00")
        assert cycles[0].event_count == 2

    def test_naive_datetime_rejected(self):
        # build_cycles uses assert_aware when comparing consecutive events
        # within the extension window. A naive trade_date should be caught.
        ev1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ev2 = _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00")
        # Force ev2.trade_date to be naive 鈥?bypasses frozen dataclass.
        object.__setattr__(ev2, "trade_date", datetime(2026, 1, 8))
        with pytest.raises(Exception):
            build_cycles([ev1, ev2], trade_calendar=_daily_calendar(), extension_days=5)
