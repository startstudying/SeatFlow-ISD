"""Tests for Direct / Spillover universe construction (Task 9)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pytest

from seatflow.dataset.event import LHBEvent
from seatflow.dataset.propagation_candidates import PropagationCandidate
from seatflow.dataset.universe import (
    DirectPool,
    PoolKind,
    SpilloverPool,
    TradeabilityState,
    UniverseEntry,
    build_direct_pool,
    build_spillover_pool,
)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _ev(
    event_id: str,
    stock_id: str,
    trade_date: str,
    *,
    available_time: Optional[str] = None,
) -> LHBEvent:
    td = _utc(trade_date)
    at = _utc(available_time) if available_time else td
    return LHBEvent(
        event_id=event_id,
        stock_id=stock_id,
        trade_date=td,
        available_time=at,
        reason_codes=["REASON_DEVIATION_7PCT"],
        total_buy=1.0e8,
        total_sell=5.0e7,
        net_amount=5.0e7,
        turnover_rate=12.5,
        event_confidence=1.0,
        source_lineage="tushare",
    )


def _tradeability_lookup(mapping: dict):
    def lookup(stock_id: str, at_time: datetime) -> TradeabilityState:
        return mapping.get(stock_id, TradeabilityState.TRADABLE)

    return lookup


class TestTradeabilityState:
    def test_tradable_value(self):
        assert TradeabilityState.TRADABLE.value == "TRADABLE"

    def test_suspended_value(self):
        assert TradeabilityState.SUSPENDED.value == "SUSPENDED"

    def test_limit_up_value(self):
        assert TradeabilityState.LIMIT_UP.value == "LIMIT_UP"

    def test_limit_down_value(self):
        assert TradeabilityState.LIMIT_DOWN.value == "LIMIT_DOWN"

    def test_delisted_value(self):
        assert TradeabilityState.DELISTED.value == "DELISTED"


class TestPoolKind:
    def test_direct_value(self):
        assert PoolKind.DIRECT.value == "DIRECT"

    def test_spillover_value(self):
        assert PoolKind.SPILLOVER.value == "SPILLOVER"


class TestUniverseEntry:
    def test_universe_entry_is_frozen(self):
        entry = UniverseEntry(
            stock_id="000001.SZ",
            cycle_id="cyc-1",
            pool=PoolKind.DIRECT,
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            exit_date=None,
            entry_reason="LHB direct entry",
            tradeability=TradeabilityState.TRADABLE,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
        )
        with pytest.raises(Exception):
            entry.stock_id = "000002.SZ"  # type: ignore[misc]

    def test_stock_id_is_string(self):
        entry = UniverseEntry(
            stock_id="000001.SZ",
            cycle_id="cyc-1",
            pool=PoolKind.DIRECT,
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            exit_date=None,
            entry_reason="LHB direct entry",
            tradeability=TradeabilityState.TRADABLE,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
        )
        assert isinstance(entry.stock_id, str)
        assert isinstance(entry.cycle_id, str)
        assert not entry.stock_id.isdigit()

    def test_timestamps_are_aware(self):
        entry = UniverseEntry(
            stock_id="000001.SZ",
            cycle_id="cyc-1",
            pool=PoolKind.DIRECT,
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            exit_date=None,
            entry_reason="LHB direct entry",
            tradeability=TradeabilityState.TRADABLE,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
        )
        assert entry.entry_date.tzinfo is not None
        assert entry.available_time.tzinfo is not None

    def test_naive_entry_date_rejected(self):
        with pytest.raises(Exception):
            UniverseEntry(
                stock_id="000001.SZ",
                cycle_id="cyc-1",
                pool=PoolKind.DIRECT,
                entry_date=datetime(2026, 1, 5),
                exit_date=None,
                entry_reason="LHB direct entry",
                tradeability=TradeabilityState.TRADABLE,
                available_time=_utc("2026-01-05T20:00:00+00:00"),
            )

    def test_naive_available_time_rejected(self):
        with pytest.raises(Exception):
            UniverseEntry(
                stock_id="000001.SZ",
                cycle_id="cyc-1",
                pool=PoolKind.DIRECT,
                entry_date=_utc("2026-01-05T00:00:00+00:00"),
                exit_date=None,
                entry_reason="LHB direct entry",
                tradeability=TradeabilityState.TRADABLE,
                available_time=datetime(2026, 1, 5, 20, 0, 0),
            )


class TestDirectPoolTradeability:
    def test_tradable_stock_added(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ok = pool.add_stock(
            "000001.SZ",
            "cyc-1",
            ev,
            tradeability=TradeabilityState.TRADABLE,
        )
        assert ok is True
        assert pool.contains("000001.SZ")
        assert len(pool) == 1

    def test_suspended_stock_excluded(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ok = pool.add_stock(
            "000001.SZ",
            "cyc-1",
            ev,
            tradeability=TradeabilityState.SUSPENDED,
        )
        assert ok is False
        assert not pool.contains("000001.SZ")
        assert len(pool) == 0

    def test_limit_up_stock_excluded(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ok = pool.add_stock(
            "000001.SZ",
            "cyc-1",
            ev,
            tradeability=TradeabilityState.LIMIT_UP,
        )
        assert ok is False
        assert len(pool) == 0

    def test_limit_down_stock_excluded(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ok = pool.add_stock(
            "000001.SZ",
            "cyc-1",
            ev,
            tradeability=TradeabilityState.LIMIT_DOWN,
        )
        assert ok is False
        assert len(pool) == 0

    def test_delisted_stock_excluded(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ok = pool.add_stock(
            "000001.SZ",
            "cyc-1",
            ev,
            tradeability=TradeabilityState.DELISTED,
        )
        assert ok is False
        assert len(pool) == 0


class TestDirectPoolStorage:
    def test_duplicate_stock_not_added_twice(self):
        pool = DirectPool()
        ev1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ev2 = _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00")
        assert pool.add_stock("000001.SZ", "cyc-1", ev1) is True
        assert pool.add_stock("000001.SZ", "cyc-1", ev2) is False
        assert len(pool) == 1

    def test_entry_carries_event_metadata(self):
        pool = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        pool.add_stock("000001.SZ", "cyc-1", ev)
        entry = pool.entries()[0]
        assert entry.pool == PoolKind.DIRECT
        assert entry.stock_id == "000001.SZ"
        assert entry.cycle_id == "cyc-1"
        assert entry.tradeability == TradeabilityState.TRADABLE

    def test_iteration_yields_entries(self):
        pool = DirectPool()
        ev1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        ev2 = _ev("e2", "000002.SZ", "2026-01-06T00:00:00+00:00")
        pool.add_stock("000001.SZ", "cyc-1", ev1)
        pool.add_stock("000002.SZ", "cyc-2", ev2)
        assert len(list(pool)) == 2
        assert {e.stock_id for e in pool} == {"000001.SZ", "000002.SZ"}

    def test_pool_is_independent_per_instance(self):
        a = DirectPool()
        b = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        a.add_stock("000001.SZ", "cyc-1", ev)
        assert a.contains("000001.SZ")
        assert not b.contains("000001.SZ")


class TestSpilloverPoolIndependence:
    def test_spillover_excludes_direct_stocks(self):
        direct = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        direct.add_stock("000001.SZ", "cyc-1", ev)

        spill = SpilloverPool()
        cand = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000999.SZ",
            target_stock_id="000001.SZ",
            relation_type="same_seat_relay",
            candidate_score=0.9,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="seat relay",
        )
        added = spill.add_candidate(cand, direct)
        assert added is False
        assert "000001.SZ" not in spill

    def test_spillover_accepts_non_direct_stock(self):
        direct = DirectPool()
        ev = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        direct.add_stock("000001.SZ", "cyc-1", ev)

        spill = SpilloverPool()
        cand = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000001.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_theme_diffusion",
            candidate_score=0.7,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="theme diffusion",
        )
        added = spill.add_candidate(cand, direct)
        assert added is True
        assert "000777.SZ" in spill

    def test_spillover_pool_rejects_duplicate_target(self):
        direct = DirectPool()
        spill = SpilloverPool()
        c1 = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000001.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_seat_relay",
            candidate_score=0.8,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="seat A",
        )
        c2 = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000002.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_theme_diffusion",
            candidate_score=0.5,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="theme A",
        )
        assert spill.add_candidate(c1, direct) is True
        assert spill.add_candidate(c2, direct) is False
        assert len(list(spill)) == 1

    def test_spillover_entry_classified_as_spillover_pool(self):
        direct = DirectPool()
        spill = SpilloverPool()
        cand = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000001.SZ",
            target_stock_id="000888.SZ",
            relation_type="capital_rotation",
            candidate_score=0.6,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="capital rotation",
        )
        spill.add_candidate(cand, direct)
        entry = list(spill)[0]
        assert entry.pool == PoolKind.SPILLOVER
        assert entry.stock_id == "000888.SZ"


class TestBuildDirectPool:
    def test_builds_pool_from_tradable_events(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000002.SZ", "2026-01-06T00:00:00+00:00"),
        ]
        lookup = _tradeability_lookup(
            {
                "000001.SZ": TradeabilityState.TRADABLE,
                "000002.SZ": TradeabilityState.TRADABLE,
            }
        )
        pool = build_direct_pool(events, lookup)
        assert len(pool) == 2
        assert {e.stock_id for e in pool} == {"000001.SZ", "000002.SZ"}

    def test_excludes_suspended_event_stocks(self):
        events = [
            _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00"),
            _ev("e2", "000002.SZ", "2026-01-06T00:00:00+00:00"),
        ]
        lookup = _tradeability_lookup(
            {
                "000001.SZ": TradeabilityState.SUSPENDED,
                "000002.SZ": TradeabilityState.TRADABLE,
            }
        )
        pool = build_direct_pool(events, lookup)
        assert len(pool) == 1
        assert pool.contains("000002.SZ")
        assert not pool.contains("000001.SZ")

    def test_excludes_limit_up_event_stocks(self):
        events = [_ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")]
        lookup = _tradeability_lookup({"000001.SZ": TradeabilityState.LIMIT_UP})
        pool = build_direct_pool(events, lookup)
        assert len(pool) == 0

    def test_unknown_tradeability_defaults_to_tradable(self):
        events = [_ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")]
        pool = build_direct_pool(events, _tradeability_lookup({}))
        assert len(pool) == 1

    def test_empty_events_yields_empty_pool(self):
        pool = build_direct_pool([], _tradeability_lookup({}))
        assert len(pool) == 0


class TestBuildSpilloverPool:
    def test_drops_targets_already_in_direct(self):
        direct = DirectPool()
        ev_direct = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        direct.add_stock("000001.SZ", "cyc-1", ev_direct)

        candidates = [
            PropagationCandidate(
                source_event_id="e1",
                source_stock_id="000001.SZ",
                target_stock_id="000001.SZ",
                relation_type="same_seat_relay",
                candidate_score=0.9,
                available_time=_utc("2026-01-05T20:00:00+00:00"),
                candidate_evidence="self",
            ),
            PropagationCandidate(
                source_event_id="e1",
                source_stock_id="000001.SZ",
                target_stock_id="000777.SZ",
                relation_type="same_theme_diffusion",
                candidate_score=0.6,
                available_time=_utc("2026-01-05T20:00:00+00:00"),
                candidate_evidence="theme A",
            ),
        ]
        spill = build_spillover_pool(direct, candidates)
        assert len(list(spill)) == 1
        assert "000777.SZ" in spill
        assert "000001.SZ" not in spill

    def test_empty_candidates_yields_empty_pool(self):
        direct = DirectPool()
        spill = build_spillover_pool(direct, [])
        assert len(list(spill)) == 0

    def test_pool_entries_carry_relation_metadata(self):
        direct = DirectPool()
        cand = PropagationCandidate(
            source_event_id="e0",
            source_stock_id="000001.SZ",
            target_stock_id="000555.SZ",
            relation_type="same_coalition_relay",
            candidate_score=0.55,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="coalition A",
        )
        spill = build_spillover_pool(direct, [cand])
        entry = list(spill)[0]
        assert entry.entry_reason.startswith("same_coalition_relay")
        assert "000555.SZ" in entry.entry_reason
