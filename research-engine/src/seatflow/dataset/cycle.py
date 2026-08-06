"""Event cycle construction (Task 8).

A *cycle* groups consecutive Dragon-Tiger-List events for the same stock
into a single attention unit. The key design property is **atomicity**:
a cycle is the split atom — the unit a model or backtest consumes — not
the individual event. The same events can be partitioned into different
cycle sets depending on ``extension_days``, which is exactly what makes
cycles the atomic unit.

Rules
-----
- Events are grouped by ``stock_id``.
- A new cycle starts at the first event for a stock.
- A subsequent event is added to the active cycle if its ``trade_date``
  is within ``extension_days`` of the active cycle's
  ``last_event_date`` (inclusive on the boundary).
- If the gap exceeds ``extension_days``, the active cycle is
  ``EXPIRED`` (its ``cycle_end`` is set to the last event's
  ``trade_date``) and a new ``ACTIVE`` cycle begins.

Determinism
-----------
Cycle IDs are derived deterministically from the stock id and the cycle's
starting event trade date, so the same input events always produce the
same cycle partition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence

from seatflow.dataset.event import LHBEvent
from seatflow.domain.time import assert_aware


# ── Enums ──────────────────────────────────────────────────────────────


class CycleStatus(str, Enum):
    """Lifecycle status of a cycle."""

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"


# ── Dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventCycle:
    """A group of consecutive LHB events for one stock.

    Attributes
    ----------
    cycle_id:
        Stable string identifier (``cyc-<stock>-<hash>``).
    stock_id:
        The stock this cycle belongs to.
    cycle_start:
        ``trade_date`` of the first event in the cycle.
    last_event_date:
        ``trade_date`` of the most recent event in the cycle.
    cycle_end:
        For an ``EXPIRED`` cycle, the ``trade_date`` of the last event
        (i.e. ``last_event_date``). ``None`` for an ``ACTIVE`` cycle.
    event_count:
        Number of events in the cycle.
    status:
        :attr:`CycleStatus.ACTIVE` or :attr:`CycleStatus.EXPIRED`.
    """

    cycle_id: str
    stock_id: str
    cycle_start: datetime
    last_event_date: datetime
    cycle_end: Optional[datetime]
    event_count: int
    status: CycleStatus


# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_EXTENSION_DAYS = 5


# ── ID generation ──────────────────────────────────────────────────────


def _cycle_id(stock_id: str, cycle_start: datetime) -> str:
    """Derive a stable cycle_id from the stock and the cycle start date.

    The ID is a string (never a bare numeric).
    """
    # Normalise the start to UTC before hashing so naive/aware differences
    # don't change the id.
    start_iso = cycle_start.astimezone(timezone.utc).isoformat()
    raw = f"{stock_id}|{start_iso}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"cyc-{stock_id}-{digest}"


# ── Window check ───────────────────────────────────────────────────────


def _sorted_calendar(trade_calendar: Sequence[datetime]) -> list[datetime]:
    calendar = sorted(set(trade_calendar))
    for day in calendar:
        assert_aware(day, "trade_calendar")
    return calendar


def _trading_day_index(calendar: Sequence[datetime], day: datetime) -> int:
    assert_aware(day, "trade_date")
    try:
        return calendar.index(day)
    except ValueError as exc:
        raise ValueError(
            f"trade_date {day.isoformat()} is not in the exchange trade_calendar"
        ) from exc


def _extension_end(
    event_date: datetime,
    calendar: Sequence[datetime],
    extension_days: int,
) -> datetime:
    end_idx = _trading_day_index(calendar, event_date) + extension_days
    if end_idx >= len(calendar):
        raise ValueError(
            f"trade_calendar does not contain {extension_days} later exchange "
            f"trading days after {event_date.isoformat()}"
        )
    return calendar[end_idx]


# ── Public API ─────────────────────────────────────────────────────────


def extend_cycle(
    cycle: EventCycle,
    new_event: LHBEvent,
    *,
    trade_calendar: Sequence[datetime],
    extension_days: int = DEFAULT_EXTENSION_DAYS,
) -> EventCycle:
    """Return a new cycle extended with ``new_event``.

    Raises ``ValueError`` if:

    - ``new_event.stock_id`` differs from ``cycle.stock_id`` (a cycle is
      per-stock), or
    - ``new_event.trade_date`` is outside the extension window of
      ``cycle.last_event_date``.

    The returned cycle keeps the same ``cycle_id`` and ``cycle_start``,
    updates ``last_event_date`` and ``event_count``, and remains
    ``ACTIVE`` (extending never expires a cycle).
    """
    if new_event.stock_id != cycle.stock_id:
        raise ValueError(
            f"Cannot extend cycle for stock {cycle.stock_id!r} with an event "
            f"for stock {new_event.stock_id!r}"
        )
    calendar = _sorted_calendar(trade_calendar)
    if cycle.cycle_end is None:
        raise ValueError("Cannot extend a cycle without a cycle_end")
    if new_event.trade_date > cycle.cycle_end:
        raise ValueError(
            f"Event {new_event.event_id!r} trade_date {new_event.trade_date} is "
            f"outside the {extension_days}-day extension window of cycle "
            f"{cycle.cycle_id!r} (cycle_end {cycle.cycle_end})"
        )
    cycle_end = _extension_end(new_event.trade_date, calendar, extension_days)
    return EventCycle(
        cycle_id=cycle.cycle_id,
        stock_id=cycle.stock_id,
        cycle_start=cycle.cycle_start,
        last_event_date=new_event.trade_date,
        cycle_end=cycle_end,
        event_count=cycle.event_count + 1,
        status=CycleStatus.ACTIVE,
    )


def build_cycles(
    events: list[LHBEvent],
    *,
    trade_calendar: Sequence[datetime],
    extension_days: int = DEFAULT_EXTENSION_DAYS,
) -> list[EventCycle]:
    """Group events into cycles per stock.

    Events are sorted by ``trade_date`` within each stock (the input need
    not be pre-sorted). Consecutive events within ``extension_days`` of
    the active cycle's last event are folded into that cycle; a larger
    gap expires the active cycle and starts a new one.

    The last cycle for each stock is left ``ACTIVE`` (``cycle_end is
    None``); earlier cycles are ``EXPIRED`` with ``cycle_end`` set to
    their ``last_event_date``.

    The returned cycles are ordered by ``(stock_id, cycle_start)``.
    """
    # Group events by stock, preserving nothing about input order — we
    # sort each group by trade_date for determinism.
    calendar = _sorted_calendar(trade_calendar)

    by_stock: dict[str, list[LHBEvent]] = {}
    for ev in events:
        by_stock.setdefault(ev.stock_id, []).append(ev)

    cycles: list[EventCycle] = []
    for stock_id in sorted(by_stock.keys()):
        stock_events = sorted(by_stock[stock_id], key=lambda e: e.trade_date)
        if not stock_events:
            continue

        # Start the first cycle for this stock.
        first = stock_events[0]
        active = EventCycle(
            cycle_id=_cycle_id(stock_id, first.trade_date),
            stock_id=stock_id,
            cycle_start=first.trade_date,
            last_event_date=first.trade_date,
            cycle_end=_extension_end(first.trade_date, calendar, extension_days),
            event_count=1,
            status=CycleStatus.ACTIVE,
        )

        for ev in stock_events[1:]:
            if active.cycle_end is not None and ev.trade_date <= active.cycle_end:
                # Same cycle — extend it.
                active = extend_cycle(
                    active,
                    ev,
                    trade_calendar=calendar,
                    extension_days=extension_days,
                )
            else:
                # Gap too large: expire the active cycle and start a new one.
                expired = EventCycle(
                    cycle_id=active.cycle_id,
                    stock_id=active.stock_id,
                    cycle_start=active.cycle_start,
                    last_event_date=active.last_event_date,
                    cycle_end=active.cycle_end,
                    event_count=active.event_count,
                    status=CycleStatus.EXPIRED,
                )
                cycles.append(expired)
                active = EventCycle(
                    cycle_id=_cycle_id(stock_id, ev.trade_date),
                    stock_id=stock_id,
                    cycle_start=ev.trade_date,
                    last_event_date=ev.trade_date,
                    cycle_end=_extension_end(ev.trade_date, calendar, extension_days),
                    event_count=1,
                    status=CycleStatus.ACTIVE,
                )

        # The final active cycle for this stock stays ACTIVE.
        cycles.append(active)

    # Stable sort by (stock_id, cycle_start) so output is deterministic
    # regardless of dict insertion order.
    cycles.sort(key=lambda c: (c.stock_id, c.cycle_start))
    return cycles
