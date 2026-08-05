"""Tests for Event extraction and PIT availability (Task 8).

Events are derived from Bronze records carrying Dragon-Tiger-List (LHB)
payloads. Each event must preserve point-in-time (PIT) semantics:

- ``available_time`` is when the system could first observe the event.
- ``available_time >= trade_date`` is enforced.
- Seat identities are resolved at the ``available_time`` of the event,
  not at ``trade_date``, so a renamed/merged seat is resolved using the
  identity that was canonical when the record became observable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from seatflow.dataset.event import (
    LHBEvent,
    SeatParticipation,
    SeatSide,
    extract_events_from_bronze,
    extract_participations_from_bronze,
)
from seatflow.identity.seat_identity import (
    SeatIdentity,
    SeatIdentityRegistry,
    SeatType,
)
from seatflow.ingestion.models import (
    BronzeRecord,
    SourceConfidence,
    compute_payload_hash,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _make_lhb_bronze(
    *,
    ts_code: str = "000001.SZ",
    trade_date: str = "20260105",
    ann_date: str | None = None,
    reason_codes: list[str] | None = None,
    total_buy: float = 1.0e8,
    total_sell: float = 5.0e7,
    net_amount: float | None = None,
    turnover_rate: float = 12.5,
    seats_buy: list[dict] | None = None,
    seats_sell: list[dict] | None = None,
    source: str = "tushare",
    source_record_id: str = "lhb-000001-20260105",
    ingestion_run_id: str = "run-001",
    source_confidence: SourceConfidence = SourceConfidence.HIGH,
) -> BronzeRecord:
    """Build a Bronze record whose raw_payload models an LHB summary row.

    The payload mirrors the Tushare top_list shape plus an embedded list of
    participating seats (buy / sell sides). ``ann_date`` doubles as the
    ``available_time`` and ``trade_date`` as the ``event_time``.
    """
    if ann_date is None:
        ann_date = trade_date  # same-day availability by default

    payload = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "ann_date": ann_date,
        "reason": ",".join(reason_codes or ["日涨幅偏离值达7%的证券"]),
        "total_buy": total_buy,
        "total_sell": total_sell,
        "net_amount": net_amount if net_amount is not None else (total_buy - total_sell),
        "turnover_rate": turnover_rate,
        "seats_buy": seats_buy or [],
        "seats_sell": seats_sell or [],
    }
    trade_dt = _parse_date(trade_date)
    ann_dt = _parse_date(ann_date)
    return BronzeRecord(
        event_time=trade_dt,
        available_time=ann_dt,
        ingested_at=_utc("2026-08-06T01:00:00+00:00"),
        source=source,
        source_record_id=source_record_id,
        source_version="tushare-pro-v1",
        revision_no=0,
        raw_payload=payload,
        raw_payload_hash=compute_payload_hash(payload),
        ingestion_run_id=ingestion_run_id,
        source_confidence=source_confidence,
    )


def _parse_date(value: str) -> datetime:
    """Parse YYYYMMDD into a tz-aware UTC datetime (midnight UTC)."""
    dt = datetime.strptime(value, "%Y%m%d")
    return dt.replace(tzinfo=timezone.utc)


def _make_registry() -> SeatIdentityRegistry:
    """Registry with a renamed seat whose validity windows do NOT overlap.

    Before 2026-01-01 the alias resolves to the old canonical name;
    from 2026-01-01 onwards it resolves to the new one. This mirrors
    the ``test_renamed_seat_resolves_historical`` pattern in
    ``tests/identity/test_seat_identity.py``.
    """
    reg = SeatIdentityRegistry()
    reg.register(
        SeatIdentity(
            seat_id="seat-ht-yitian",
            canonical_name="华泰证券深圳益田路荣超商务中心",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            valid_to=_utc("2025-12-31T23:59:59+00:00"),
            aliases=("华泰益田路",),
            identity_confidence=0.95,
            source_evidence="manual",
        )
    )
    reg.register(
        SeatIdentity(
            seat_id="seat-ht-yitian",
            canonical_name="华泰证券深圳益田路(改名后)",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2026-01-01T00:00:00+00:00"),
            aliases=("华泰益田路",),
            identity_confidence=0.95,
            source_evidence="manual",
        )
    )
    return reg


# ── Event extraction ───────────────────────────────────────────────────


class TestExtractEvents:
    def test_extracts_single_event(self):
        bronze = _make_lhb_bronze(
            reason_codes=["日涨幅偏离值达7%的证券"],
        )
        events = extract_events_from_bronze([bronze], _make_registry())
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, LHBEvent)
        assert ev.stock_id == "000001.SZ"
        assert ev.trade_date == _parse_date("20260105")
        assert ev.total_buy == pytest.approx(1.0e8)
        assert ev.total_sell == pytest.approx(5.0e7)
        assert ev.net_amount == pytest.approx(5.0e7)
        assert ev.turnover_rate == pytest.approx(12.5)
        assert ev.reason_codes == ["日涨幅偏离值达7%的证券"]
        assert ev.source_lineage == "tushare"

    def test_event_id_is_string_and_stable(self):
        bronze = _make_lhb_bronze(source_record_id="lhb-000001-20260105")
        events = extract_events_from_bronze([bronze], _make_registry())
        assert isinstance(events[0].event_id, str)
        # Re-extracting the same bronze yields the same event_id
        again = extract_events_from_bronze([bronze], _make_registry())
        assert again[0].event_id == events[0].event_id

    def test_multiple_bronze_records_yield_multiple_events(self):
        b1 = _make_lhb_bronze(ts_code="000001.SZ", source_record_id="r1")
        b2 = _make_lhb_bronze(
            ts_code="000002.SZ", source_record_id="r2", trade_date="20260106"
        )
        events = extract_events_from_bronze([b1, b2], _make_registry())
        assert len(events) == 2
        assert {e.stock_id for e in events} == {"000001.SZ", "000002.SZ"}

    def test_empty_input_yields_empty_list(self):
        assert extract_events_from_bronze([], _make_registry()) == []

    def test_event_confidence_reflects_source_confidence(self):
        bronze = _make_lhb_bronze(source_confidence=SourceConfidence.HIGH)
        events = extract_events_from_bronze([bronze], _make_registry())
        assert events[0].event_confidence == pytest.approx(1.0)

        bronze_low = _make_lhb_bronze(source_confidence=SourceConfidence.LOW)
        events_low = extract_events_from_bronze([bronze_low], _make_registry())
        assert events_low[0].event_confidence < 1.0

    def test_source_lineage_tracks_origin(self):
        bronze = _make_lhb_bronze(source="akshare", source_record_id="ak-1")
        events = extract_events_from_bronze([bronze], _make_registry())
        assert events[0].source_lineage == "akshare"


# ── PIT availability ordering ───────────────────────────────────────────


class TestAvailabilityOrdering:
    def test_available_time_at_or_after_trade_date(self):
        # ann_date == trade_date: same-day availability is allowed.
        bronze = _make_lhb_bronze(trade_date="20260105", ann_date="20260105")
        events = extract_events_from_bronze([bronze], _make_registry())
        assert events[0].available_time >= events[0].trade_date

    def test_available_time_after_trade_date(self):
        bronze = _make_lhb_bronze(trade_date="20260105", ann_date="20260107")
        events = extract_events_from_bronze([bronze], _make_registry())
        assert events[0].available_time > events[0].trade_date

    def test_available_time_before_trade_date_raises(self):
        # Pre-announcement violates PIT semantics for LHB events.
        bronze = _make_lhb_bronze(trade_date="20260107", ann_date="20260105")
        with pytest.raises(ValueError, match="available_time"):
            extract_events_from_bronze([bronze], _make_registry())


# ── PIT seat identity resolution ───────────────────────────────────────


class TestSeatIdentityResolution:
    def test_seat_resolved_at_available_time_not_trade_date(self):
        """Seat is resolved at available_time so a renamed seat resolves to the
        identity canonical *when the record became observable*.

        In this scenario the seat alias "华泰益田路" was valid under one
        canonical name before 2026-01-01 and another after. The event's
        trade_date is before the rename but the available_time is after, so
        the resolved canonical name must reflect the post-rename identity.
        """
        bronze = _make_lhb_bronze(
            trade_date="20251230",  # before rename
            ann_date="20260102",    # after rename (>= trade_date)
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 3.0e7},
            ],
        )
        parts = extract_participations_from_bronze([bronze], _make_registry())
        assert len(parts) == 1
        # The participation's seat_id should resolve via the post-rename identity.
        assert parts[0].seat_id == "seat-ht-yitian"
        # identity_confidence carried from the registry.
        assert parts[0].identity_confidence == pytest.approx(0.95)

    def test_seat_outside_validity_period_raises(self):
        """If the alias is known but outside its validity window at
        available_time, a KeyError propagates (no silent UNKNOWN fallback)."""
        reg = SeatIdentityRegistry()
        reg.register(
            SeatIdentity(
                seat_id="seat-x",
                canonical_name="临时席位",
                seat_type=SeatType.BROKER_BRANCH,
                valid_from=_utc("2026-06-01T00:00:00+00:00"),
                valid_to=_utc("2026-06-30T00:00:00+00:00"),
                aliases=("临时席位",),
            )
        )
        bronze = _make_lhb_bronze(
            trade_date="20260105",
            ann_date="20260105",
            seats_buy=[{"name": "临时席位", "rank": 1, "buy_amount": 1.0e7}],
        )
        with pytest.raises(KeyError):
            extract_participations_from_bronze([bronze], reg)

    def test_unregistered_seat_yields_unknown_identity(self):
        reg = SeatIdentityRegistry()
        bronze = _make_lhb_bronze(
            seats_buy=[{"name": "从未登记过的席位", "rank": 1, "buy_amount": 1.0e7}],
        )
        parts = extract_participations_from_bronze([bronze], reg)
        assert len(parts) == 1
        assert parts[0].identity_confidence == pytest.approx(0.0)
        assert parts[0].seat_id.startswith("UNKNOWN-")


# ── Participation extraction ───────────────────────────────────────────


class TestExtractParticipations:
    def test_buy_and_sell_participations_extracted(self):
        bronze = _make_lhb_bronze(
            total_buy=1.0e8,
            total_sell=5.0e7,
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 6.0e7, "sell_amount": 0.0},
                {"name": "中信上海溧阳路", "rank": 2, "buy_amount": 4.0e7, "sell_amount": 0.0},
            ],
            seats_sell=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 0.0, "sell_amount": 3.0e7},
                {"name": "国泰君安上海江苏路", "rank": 2, "buy_amount": 0.0, "sell_amount": 2.0e7},
            ],
        )
        reg = _make_registry()
        parts = extract_participations_from_bronze([bronze], reg)
        assert len(parts) == 4

        buy_parts = [p for p in parts if p.side == SeatSide.BUY]
        sell_parts = [p for p in parts if p.side == SeatSide.SELL]
        assert len(buy_parts) == 2
        assert len(sell_parts) == 2

    def test_participation_carries_event_id(self):
        bronze = _make_lhb_bronze(
            seats_buy=[{"name": "华泰益田路", "rank": 1, "buy_amount": 1.0e7}],
        )
        events = extract_events_from_bronze([bronze], _make_registry())
        parts = extract_participations_from_bronze([bronze], _make_registry())
        assert all(p.event_id == events[0].event_id for p in parts)

    def test_participation_amount_share_computed(self):
        bronze = _make_lhb_bronze(
            total_buy=1.0e8,
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 6.0e7, "sell_amount": 0.0},
                {"name": "中信上海溧阳路", "rank": 2, "buy_amount": 4.0e7, "sell_amount": 0.0},
            ],
        )
        parts = extract_participations_from_bronze([bronze], _make_registry())
        buy_parts = sorted(
            [p for p in parts if p.side == SeatSide.BUY], key=lambda p: p.rank
        )
        assert buy_parts[0].amount_share == pytest.approx(0.6)
        assert buy_parts[1].amount_share == pytest.approx(0.4)

    def test_participation_net_amount_computed(self):
        bronze = _make_lhb_bronze(
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 6.0e7, "sell_amount": 1.0e7},
            ],
        )
        parts = extract_participations_from_bronze([bronze], _make_registry())
        assert parts[0].net_amount == pytest.approx(5.0e7)

    def test_participation_available_time_matches_event(self):
        bronze = _make_lhb_bronze(
            trade_date="20260105",
            ann_date="20260106",
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 1.0e7},
            ],
        )
        events = extract_events_from_bronze([bronze], _make_registry())
        parts = extract_participations_from_bronze([bronze], _make_registry())
        assert parts[0].available_time == events[0].available_time

    def test_participation_rank_preserved(self):
        bronze = _make_lhb_bronze(
            seats_buy=[
                {"name": "华泰益田路", "rank": 1, "buy_amount": 1.0e7},
                {"name": "中信上海溧阳路", "rank": 3, "buy_amount": 5.0e6},
            ],
        )
        parts = extract_participations_from_bronze([bronze], _make_registry())
        buy_parts = [p for p in parts if p.side == SeatSide.BUY]
        ranks = sorted(p.rank for p in buy_parts)
        assert ranks == [1, 3]

    def test_participations_empty_when_no_seats(self):
        bronze = _make_lhb_bronze(seats_buy=[], seats_sell=[])
        parts = extract_participations_from_bronze([bronze], _make_registry())
        assert parts == []


# ── Dataclass contracts ────────────────────────────────────────────────


class TestDataclassContracts:
    def test_lhb_event_is_frozen(self):
        bronze = _make_lhb_bronze()
        events = extract_events_from_bronze([bronze], _make_registry())
        with pytest.raises(Exception):
            events[0].total_buy = 0.0  # type: ignore[misc]

    def test_seat_participation_is_frozen(self):
        bronze = _make_lhb_bronze(
            seats_buy=[{"name": "华泰益田路", "rank": 1, "buy_amount": 1.0e7}],
        )
        parts = extract_participations_from_bronze([bronze], _make_registry())
        with pytest.raises(Exception):
            parts[0].buy_amount = 0.0  # type: ignore[misc]

    def test_seat_side_enum_values(self):
        assert SeatSide.BUY.value == "BUY"
        assert SeatSide.SELL.value == "SELL"
