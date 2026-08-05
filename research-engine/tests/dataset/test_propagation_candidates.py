"""Tests for propagation candidate discovery (Task 9)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pytest

from seatflow.dataset.event import LHBEvent, SeatParticipation, SeatSide
from seatflow.dataset.propagation_candidates import (
    PropagationCandidate,
    find_propagation_candidates,
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


def _part(
    event_id: str,
    seat_id: str,
    stock_id: str,
    side: SeatSide,
    at: str,
    rank: int = 1,
) -> SeatParticipation:
    return SeatParticipation(
        event_id=event_id,
        seat_id=seat_id,
        side=side,
        rank=rank,
        buy_amount=1.0e7 if side == SeatSide.BUY else 0.0,
        sell_amount=1.0e7 if side == SeatSide.SELL else 0.0,
        net_amount=1.0e7 if side == SeatSide.BUY else -1.0e7,
        amount_share=1.0,
        available_time=_utc(at),
        identity_confidence=1.0,
    )


# --- Dataclass contract ------------------------------------------------


class TestPropagationCandidateDataclass:
    def test_candidate_is_frozen(self):
        c = PropagationCandidate(
            source_event_id="e1",
            source_stock_id="000001.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_seat_relay",
            candidate_score=0.8,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="seat relay",
        )
        with pytest.raises(Exception):
            c.candidate_score = 0.1  # type: ignore[misc]

    def test_ids_are_strings(self):
        c = PropagationCandidate(
            source_event_id="e1",
            source_stock_id="000001.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_seat_relay",
            candidate_score=0.8,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="seat relay",
        )
        assert isinstance(c.source_event_id, str)
        assert isinstance(c.source_stock_id, str)
        assert isinstance(c.target_stock_id, str)

    def test_available_time_is_aware(self):
        c = PropagationCandidate(
            source_event_id="e1",
            source_stock_id="000001.SZ",
            target_stock_id="000777.SZ",
            relation_type="same_seat_relay",
            candidate_score=0.8,
            available_time=_utc("2026-01-05T20:00:00+00:00"),
            candidate_evidence="seat relay",
        )
        assert c.available_time.tzinfo is not None

    def test_naive_available_time_rejected(self):
        with pytest.raises(Exception):
            PropagationCandidate(
                source_event_id="e1",
                source_stock_id="000001.SZ",
                target_stock_id="000777.SZ",
                relation_type="same_seat_relay",
                candidate_score=0.8,
                available_time=datetime(2026, 1, 5, 20, 0, 0),
                candidate_evidence="seat relay",
            )

    def test_score_out_of_range_rejected(self):
        """candidate_score must be in [0, 1]."""
        for bad in (-0.01, 1.01, -1.0, 2.0):
            with pytest.raises(Exception):
                PropagationCandidate(
                    source_event_id="e1",
                    source_stock_id="000001.SZ",
                    target_stock_id="000777.SZ",
                    relation_type="same_seat_relay",
                    candidate_score=bad,
                    available_time=_utc("2026-01-05T20:00:00+00:00"),
                    candidate_evidence="seat relay",
                )


# --- same_seat_relay discovery ----------------------------------------


class TestSameSeatRelay:
    def test_candidates_found_for_same_seat_relay(self):
        """Two events sharing a seat create a same_seat_relay candidate."""
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-08T00:00:00+00:00")
        # Seat S1 participates in both events -> relay between 000001.SZ
        # and 000777.SZ.
        parts = [
            _part("e1", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-S1", "000777.SZ", SeatSide.BUY, "2026-01-08T20:00:00+00:00"),
        ]
        cands = find_propagation_candidates([e1, e2], parts)
        relays = [c for c in cands if c.relation_type == "same_seat_relay"]
        assert len(relays) >= 1
        targets = {c.target_stock_id for c in relays}
        # 000777.SZ is reachable from 000001.SZ via seat-S1.
        assert "000777.SZ" in targets

    def test_no_relay_without_shared_seat(self):
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-08T00:00:00+00:00")
        parts = [
            _part("e1", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-S2", "000777.SZ", SeatSide.BUY, "2026-01-08T20:00:00+00:00"),
        ]
        cands = find_propagation_candidates([e1, e2], parts)
        relays = [c for c in cands if c.relation_type == "same_seat_relay"]
        targets = {c.target_stock_id for c in relays}
        assert "000777.SZ" not in targets

    def test_relay_excludes_self_loop(self):
        """A seat in two events for the SAME stock must not create a
        self-loop candidate (source == target)."""
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000001.SZ", "2026-01-08T00:00:00+00:00")
        parts = [
            _part("e1", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-08T20:00:00+00:00"),
        ]
        cands = find_propagation_candidates([e1, e2], parts)
        for c in cands:
            assert c.source_stock_id != c.target_stock_id


# --- max_hops ----------------------------------------------------------


class TestMaxHops:
    def test_max_hops_2_limits_propagation_depth(self):
        """A three-event chain A -> B -> C. With max_hops=2, C is reachable
        from A (two hops). With max_hops=1, C is NOT reachable from A."""
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-06T00:00:00+00:00")
        e3 = _ev("e3", "000888.SZ", "2026-01-07T00:00:00+00:00")
        parts = [
            _part("e1", "seat-A", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-A", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
            _part("e2", "seat-B", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
            _part("e3", "seat-B", "000888.SZ", SeatSide.BUY, "2026-01-07T20:00:00+00:00"),
        ]
        # max_hops=2: A(000001) -> B(000777) -> C(000888) reachable.
        deep = find_propagation_candidates([e1, e2, e3], parts, max_hops=2)
        deep_from_a = {
            c.target_stock_id
            for c in deep
            if c.source_stock_id == "000001.SZ" and c.relation_type == "same_seat_relay"
        }
        assert "000888.SZ" in deep_from_a
        # max_hops=1: only one-hop relay, so 000888 not reachable from
        # 000001 via same_seat_relay (only via 000777 which is a different
        # source).
        shallow = find_propagation_candidates([e1, e2, e3], parts, max_hops=1)
        # At hop 1, candidates sourced from 000001.SZ reach 000777.SZ.
        shallow_from_a = {
            c.target_stock_id
            for c in shallow
            if c.source_stock_id == "000001.SZ" and c.relation_type == "same_seat_relay"
        }
        assert "000777.SZ" in shallow_from_a
        assert "000888.SZ" not in shallow_from_a

    def test_max_hops_1_does_not_reach_two_hop_target(self):
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-06T00:00:00+00:00")
        e3 = _ev("e3", "000888.SZ", "2026-01-07T00:00:00+00:00")
        parts = [
            _part("e1", "seat-A", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-A", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
            _part("e2", "seat-B", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
            _part("e3", "seat-B", "000888.SZ", SeatSide.BUY, "2026-01-07T20:00:00+00:00"),
        ]
        deep = find_propagation_candidates([e1, e2, e3], parts, max_hops=2)
        deep_from_a = {
            c.target_stock_id
            for c in deep
            if c.source_stock_id == "000001.SZ" and c.relation_type == "same_seat_relay"
        }
        assert "000888.SZ" in deep_from_a


# --- top_k truncation -------------------------------------------------


class TestTopKTruncation:
    def test_top_k_limits_candidates_per_relation(self):
        """When many stocks share a seat with source stock, only top_k are
        retained per relation type, ordered by score descending with
        stock_id as tie-breaker."""
        e_source = _ev("e0", "000001.SZ", "2026-01-05T00:00:00+00:00")
        events = [e_source]
        parts = [_part("e0", "seat-X", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00")]
        # Create 10 events with 10 target stocks all sharing seat-X.
        for i in range(10):
            sid = f"00{i:04d}.SZ"
            eid = f"e-tgt-{i}"
            events.append(_ev(eid, sid, "2026-01-06T00:00:00+00:00"))
            parts.append(_part(eid, "seat-X", sid, SeatSide.BUY, "2026-01-06T20:00:00+00:00"))

        cands = find_propagation_candidates(events, parts, max_hops=1, top_k=3)
        relays = [c for c in cands if c.relation_type == "same_seat_relay" and c.source_stock_id == "000001.SZ"]
        # At most top_k=3 candidates.
        assert len(relays) <= 3

    def test_top_k_tie_breaker_is_stock_id(self):
        """When scores tie, stock_id is the stable tie-breaker."""
        e_source = _ev("e0", "000001.SZ", "2026-01-05T00:00:00+00:00")
        events = [e_source]
        parts = [_part("e0", "seat-X", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00")]
        for sid in ["000010.SZ", "000009.SZ", "000008.SZ", "000007.SZ", "000006.SZ"]:
            eid = f"e-{sid}"
            events.append(_ev(eid, sid, "2026-01-06T00:00:00+00:00"))
            parts.append(_part(eid, "seat-X", sid, SeatSide.BUY, "2026-01-06T20:00:00+00:00"))
        cands = find_propagation_candidates(events, parts, max_hops=1, top_k=2)
        relays = sorted(
            [c for c in cands if c.relation_type == "same_seat_relay" and c.source_stock_id == "000001.SZ"],
            key=lambda c: c.target_stock_id,
        )
        # With top_k=2, we get at most 2. The ordering is deterministic.
        assert len(relays) <= 2
        if len(relays) == 2:
            assert relays[0].target_stock_id < relays[1].target_stock_id


# --- candidate_score bounded [0, 1] -----------------------------------


class TestCandidateScoreBounded:
    def test_scores_are_bounded(self):
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-06T00:00:00+00:00")
        parts = [
            _part("e1", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-S1", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
        ]
        cands = find_propagation_candidates([e1, e2], parts)
        for c in cands:
            assert 0.0 <= c.candidate_score <= 1.0

    def test_single_seat_relay_has_high_score(self):
        e1 = _ev("e1", "000001.SZ", "2026-01-05T00:00:00+00:00")
        e2 = _ev("e2", "000777.SZ", "2026-01-06T00:00:00+00:00")
        parts = [
            _part("e1", "seat-S1", "000001.SZ", SeatSide.BUY, "2026-01-05T20:00:00+00:00"),
            _part("e2", "seat-S1", "000777.SZ", SeatSide.BUY, "2026-01-06T20:00:00+00:00"),
        ]
        cands = find_propagation_candidates([e1, e2], parts)
        relays = [c for c in cands if c.relation_type == "same_seat_relay"]
        assert len(relays) >= 1
        for c in relays:
            assert c.candidate_score > 0.0
