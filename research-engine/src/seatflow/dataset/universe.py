"""Direct / Spillover universe construction (Task 9).

The universe module partitions stocks into two **independently
constructed** pools:

- **Direct pool**: stocks directly listed on the LHB at prediction time.
  Only ``TRADABLE`` stocks are admitted.
- **Spillover pool**: stocks not directly on the LHB but connected via
  propagation relations. A stock already in the Direct pool is
  **forbidden** from the Spillover pool.

Each pool entry is a :class:`UniverseEntry` that carries its pool
classification, tradeability state, and PIT timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from seatflow.dataset.event import LHBEvent
from seatflow.dataset.propagation_candidates import PropagationCandidate
from seatflow.domain.time import assert_aware


# --- Enums -------------------------------------------------------------


class TradeabilityState(str, Enum):
    """Tradeability classification of a stock at a point in time."""

    TRADABLE = "TRADABLE"
    SUSPENDED = "SUSPENDED"
    LIMIT_UP = "LIMIT_UP"
    LIMIT_DOWN = "LIMIT_DOWN"
    DELISTED = "DELISTED"


class PoolKind(str, Enum):
    """Pool classification for a universe entry."""

    DIRECT = "DIRECT"
    SPILLOVER = "SPILLOVER"


# --- UniverseEntry ----------------------------------------------------


@dataclass(frozen=True)
class UniverseEntry:
    """A single stock entry in a universe pool.

    Attributes
    ----------
    stock_id:
        The stock ticker string (e.g. ``"000001.SZ"``).
    cycle_id:
        The cycle this entry belongs to.
    pool:
        :attr:`PoolKind.DIRECT` or :attr:`PoolKind.SPILLOVER`.
    entry_date:
        When the stock entered the pool (PIT, tz-aware UTC).
    exit_date:
        When the stock left the pool, or ``None`` if still active.
    entry_reason:
        Human-readable reason for admission.
    tradeability:
        :class:`TradeabilityState` at entry.
    available_time:
        When the entry first became observable (PIT, tz-aware UTC).
    """

    stock_id: str
    cycle_id: str
    pool: PoolKind
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_reason: str
    tradeability: TradeabilityState
    available_time: datetime

    def __post_init__(self) -> None:
        assert_aware(self.entry_date, "entry_date")
        assert_aware(self.available_time, "available_time")
        if self.exit_date is not None:
            assert_aware(self.exit_date, "exit_date")


# --- Tradeability lookup type ----------------------------------------


TradeabilityLookup = Callable[[str, datetime], TradeabilityState]
"""A function that returns the tradeability state of a stock at a given
point in time."""


# --- DirectPool -------------------------------------------------------


class DirectPool:
    """Pool of stocks directly listed on the LHB.

    Only ``TRADABLE`` stocks are admitted; any other tradeability state
    (SUSPENDED, LIMIT_UP, LIMIT_DOWN, DELISTED) causes the stock to be
    rejected. A stock cannot be added twice.
    """

    def __init__(self) -> None:
        self._entries: dict[str, UniverseEntry] = {}

    def add_stock(
        self,
        stock_id: str,
        cycle_id: str,
        event: LHBEvent,
        *,
        tradeability: TradeabilityState = TradeabilityState.TRADABLE,
    ) -> bool:
        """Attempt to add ``stock_id`` to the Direct pool.

        Returns ``True`` if admitted, ``False`` if rejected (untradable or
        already present).
        """
        if tradeability != TradeabilityState.TRADABLE:
            return False
        if stock_id in self._entries:
            return False
        self._entries[stock_id] = UniverseEntry(
            stock_id=stock_id,
            cycle_id=cycle_id,
            pool=PoolKind.DIRECT,
            entry_date=event.trade_date,
            exit_date=None,
            entry_reason=f"LHB direct entry ({event.event_id})",
            tradeability=tradeability,
            available_time=event.available_time,
        )
        return True

    def contains(self, stock_id: str) -> bool:
        return stock_id in self._entries

    def __contains__(self, stock_id: str) -> bool:
        return stock_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def entries(self) -> list[UniverseEntry]:
        return list(self._entries.values())


# --- SpilloverPool ----------------------------------------------------


class SpilloverPool:
    """Pool of stocks reached via propagation relations.

    A stock already in the Direct pool is **forbidden** from this pool.
    A target stock cannot be added twice.
    """

    def __init__(self) -> None:
        self._entries: dict[str, UniverseEntry] = {}

    def add_candidate(
        self,
        candidate: PropagationCandidate,
        direct_pool: DirectPool,
    ) -> bool:
        """Attempt to add a propagation candidate's target stock.

        Returns ``True`` if admitted, ``False`` if rejected (target is in
        the Direct pool, or already present).
        """
        target = candidate.target_stock_id
        if direct_pool.contains(target):
            return False
        if target in self._entries:
            return False
        self._entries[target] = UniverseEntry(
            stock_id=target,
            cycle_id=candidate.source_event_id,
            pool=PoolKind.SPILLOVER,
            entry_date=candidate.available_time,
            exit_date=None,
            entry_reason=f"{candidate.relation_type} -> {target} (score={candidate.candidate_score:.2f})",
            tradeability=TradeabilityState.TRADABLE,
            available_time=candidate.available_time,
        )
        return True

    def __contains__(self, stock_id: str) -> bool:
        return stock_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def entries(self) -> list[UniverseEntry]:
        return list(self._entries.values())


# --- Builder functions ------------------------------------------------


def build_direct_pool(
    events: list[LHBEvent],
    tradeability_lookup: TradeabilityLookup,
) -> DirectPool:
    """Build a Direct pool from LHB events.

    Each event's stock is looked up for tradeability. Only ``TRADABLE``
    stocks are admitted. Stocks not found in the lookup default to
    ``TRADABLE``.
    """
    pool = DirectPool()
    for ev in events:
        tb = tradeability_lookup(ev.stock_id, ev.available_time)
        pool.add_stock(ev.stock_id, f"cyc-{ev.stock_id}", ev, tradeability=tb)
    return pool


def build_spillover_pool(
    direct_pool: DirectPool,
    candidates: list[PropagationCandidate],
) -> SpilloverPool:
    """Build a Spillover pool from propagation candidates.

    Candidates whose target is already in the Direct pool are silently
    dropped. The Spillover pool is independently constructed from the
    Direct pool.
    """
    pool = SpilloverPool()
    for cand in candidates:
        pool.add_candidate(cand, direct_pool)
    return pool
