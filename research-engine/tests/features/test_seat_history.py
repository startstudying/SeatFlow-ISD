"""Tests for PIT seat history computation (Task 10)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from seatflow.dataset.event import SeatParticipation, SeatSide
from seatflow.features.seat_history import SeatHistory, compute_seat_history


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _part(
    event_id: str,
    seat_id: str,
    side: SeatSide,
    buy_amount: float,
    sell_amount: float,
    at: str,
) -> SeatParticipation:
    return SeatParticipation(
        event_id=event_id,
        seat_id=seat_id,
        side=side,
        rank=1,
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        net_amount=buy_amount - sell_amount,
        amount_share=1.0,
        available_time=_utc(at),
        identity_confidence=1.0,
    )


def _label(
    mature_time: str,
    mature_return: float,
) -> dict:
    """A minimal mature label dict with mature_time and mature_return."""
    return {"mature_time": _utc(mature_time), "mature_return": mature_return}


# --- SeatHistory dataclass -------------------------------------------


class TestSeatHistoryDataclass:
    def test_seat_history_is_frozen(self):
        hist = SeatHistory(
            seat_id="s1",
            snapshot_time=_utc("2026-01-10T00:00:00+00:00"),
            lhb_count_5d=3,
            lhb_count_20d=10,
            buy_amount_5d=1.0e8,
            sell_amount_5d=5.0e7,
            net_amount_5d=5.0e7,
            mature_return_mean=0.05,
            win_rate=0.6,
            event_reliability=0.8,
        )
        with pytest.raises(Exception):
            hist.lhb_count_5d = 0  # type: ignore[misc]

    def test_seat_history_ids_are_strings(self):
        hist = SeatHistory(
            seat_id="s1",
            snapshot_time=_utc("2026-01-10T00:00:00+00:00"),
            lhb_count_5d=1,
            lhb_count_20d=2,
            buy_amount_5d=1.0,
            sell_amount_5d=0.5,
            net_amount_5d=0.5,
            mature_return_mean=0.0,
            win_rate=0.0,
            event_reliability=0.0,
        )
        assert isinstance(hist.seat_id, str)

    def test_snapshot_time_is_aware(self):
        hist = SeatHistory(
            seat_id="s1",
            snapshot_time=_utc("2026-01-10T00:00:00+00:00"),
            lhb_count_5d=1,
            lhb_count_20d=2,
            buy_amount_5d=1.0,
            sell_amount_5d=0.5,
            net_amount_5d=0.5,
            mature_return_mean=0.0,
            win_rate=0.0,
            event_reliability=0.0,
        )
        assert hist.snapshot_time.tzinfo is not None


# --- PIT participation filtering -------------------------------------


class TestPitParticipationFiltering:
    def test_only_pit_participations_used(self):
        """A participation with available_time > at_time must be excluded."""
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
            _part("e2", "s1", SeatSide.BUY, 2.0e7, 0.0, "2026-01-08T00:00:00+00:00"),
        ]
        labels = []
        hist = compute_seat_history("s1", _utc("2026-01-07T00:00:00+00:00"), parts, labels)
        # Only e1 (avail 01-05) is within PIT; e2 (avail 01-08) is excluded.
        assert hist.lhb_count_5d == 1
        assert hist.buy_amount_5d == pytest.approx(1.0e7)

    def test_participation_exactly_at_at_time_included(self):
        """available_time == at_time is included (boundary inclusive)."""
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        hist = compute_seat_history("s1", _utc("2026-01-05T00:00:00+00:00"), parts, [])
        assert hist.lhb_count_5d == 1

    def test_empty_participations(self):
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), [], [])
        assert hist.lhb_count_5d == 0
        assert hist.buy_amount_5d == 0.0


# --- Immature label exclusion ----------------------------------------


class TestImmatureLabelExclusion:
    def test_immature_labels_excluded(self):
        """A label whose mature_time > at_time must be excluded."""
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-04T00:00:00+00:00", 0.10),  # mature
            _label("2026-01-12T00:00:00+00:00", 0.50),  # immature at 01-10
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        # Only the first label is mature at at_time.
        assert hist.mature_return_mean == pytest.approx(0.10)

    def test_label_exactly_at_at_time_included(self):
        """mature_time == at_time is included (boundary inclusive)."""
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-10T00:00:00+00:00", 0.20),
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        assert hist.mature_return_mean == pytest.approx(0.20)

    def test_no_mature_labels(self):
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-12T00:00:00+00:00", 0.50),  # immature
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        assert hist.mature_return_mean == 0.0
        assert hist.win_rate == 0.0


# --- mature_return_mean computation ----------------------------------


class TestMatureReturnMean:
    def test_mean_of_mature_returns(self):
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-04T00:00:00+00:00", 0.02),
            _label("2026-01-05T00:00:00+00:00", 0.04),
            _label("2026-01-06T00:00:00+00:00", 0.06),
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        # Mean of 0.02, 0.04, 0.06 = 0.04
        assert hist.mature_return_mean == pytest.approx(0.04)

    def test_mean_handles_negative_returns(self):
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-04T00:00:00+00:00", 0.10),
            _label("2026-01-05T00:00:00+00:00", -0.10),
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        assert hist.mature_return_mean == pytest.approx(0.0)


# --- win_rate computation --------------------------------------------


class TestWinRate:
    def test_win_rate_counted_over_positive_returns(self):
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        labels = [
            _label("2026-01-04T00:00:00+00:00", 0.05),  # win
            _label("2026-01-05T00:00:00+00:00", -0.05),  # loss
            _label("2026-01-06T00:00:00+00:00", 0.02),  # win
            _label("2026-01-07T00:00:00+00:00", 0.0),  # not a win
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, labels)
        # 2 wins out of 4 mature labels = 0.5
        assert hist.win_rate == pytest.approx(0.5)

    def test_win_rate_with_no_labels_is_zero(self):
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        hist = compute_seat_history("s1", _utc("2026-01-10T00:00:00+00:00"), parts, [])
        assert hist.win_rate == 0.0


# --- 5d/20d window aggregation ---------------------------------------


class TestWindowAggregation:
    def test_5d_count_excludes_events_outside_5d_window(self):
        at = _utc("2026-01-10T00:00:00+00:00")
        # Events at varying days back from at_time
        parts = [
            _part("e0", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-10T00:00:00+00:00"),  # 0d
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-09T00:00:00+00:00"),  # 1d
            _part("e2", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-08T00:00:00+00:00"),  # 2d
            _part("e3", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-07T00:00:00+00:00"),  # 3d
            _part("e4", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-06T00:00:00+00:00"),  # 4d
            _part("e5", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),  # 5d -- in 5d window
            _part("e6", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-04T00:00:00+00:00"),  # 6d -- out
        ]
        hist = compute_seat_history("s1", at, parts, [])
        # 5d window: 0..5 days back, inclusive. 6 events.
        assert hist.lhb_count_5d == 6
        assert hist.buy_amount_5d == pytest.approx(6.0e7)

    def test_20d_count_includes_more_events(self):
        at = _utc("2026-01-10T00:00:00+00:00")
        parts = []
        for d in range(25):
            ts = (at - timedelta(days=d)).isoformat()
            parts.append(_part(f"e{d}", "s1", SeatSide.BUY, 1.0e6, 0.0, ts))
        hist = compute_seat_history("s1", at, parts, [])
        # 20d window: 0..20 days back = 21 events
        assert hist.lhb_count_20d == 21
        # 5d window: 0..5 days back = 6 events
        assert hist.lhb_count_5d == 6

    def test_buy_sell_net_amounts_aggregated(self):
        at = _utc("2026-01-10T00:00:00+00:00")
        parts = [
            _part("e1", "s1", SeatSide.BUY, 3.0e7, 0.0, "2026-01-08T00:00:00+00:00"),
            _part("e2", "s1", SeatSide.SELL, 0.0, 1.0e7, "2026-01-09T00:00:00+00:00"),
        ]
        hist = compute_seat_history("s1", at, parts, [])
        assert hist.buy_amount_5d == pytest.approx(3.0e7)
        assert hist.sell_amount_5d == pytest.approx(1.0e7)
        assert hist.net_amount_5d == pytest.approx(2.0e7)

    def test_only_matching_seat_participations_counted(self):
        """Participations for a different seat_id must be excluded."""
        at = _utc("2026-01-10T00:00:00+00:00")
        parts = [
            _part("e1", "s1", SeatSide.BUY, 1.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
            _part("e2", "OTHER", SeatSide.BUY, 9.0e7, 0.0, "2026-01-05T00:00:00+00:00"),
        ]
        hist = compute_seat_history("s1", at, parts, [])
        assert hist.lhb_count_5d == 1
        assert hist.buy_amount_5d == pytest.approx(1.0e7)
