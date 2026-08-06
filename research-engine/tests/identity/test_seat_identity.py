"""Tests for PIT seat identity resolution (Task 7)."""

from datetime import datetime

import pytest

from seatflow.identity.seat_identity import (
    SeatIdentity,
    SeatIdentityRegistry,
    SeatType,
)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestSeatIdentityResolution:
    def test_resolve_by_canonical_name(self):
        reg = SeatIdentityRegistry()
        ident = SeatIdentity(
            seat_id="seat-001",
            canonical_name="华泰证券股份有限公司深圳益田路荣超商务中心证券营业部",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            valid_to=None,
            aliases=("华泰深圳益田路",),
        )
        reg.register(ident)
        result = reg.resolve_seat(
            "华泰证券股份有限公司深圳益田路荣超商务中心证券营业部",
            _utc("2025-01-01T00:00:00+00:00"),
        )
        assert result.seat_id == "seat-001"
        assert result.resolved is True

    def test_resolve_by_alias(self):
        reg = SeatIdentityRegistry()
        ident = SeatIdentity(
            seat_id="seat-001",
            canonical_name="华泰证券深圳益田路",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            aliases=("华泰益田", "益田路"),
        )
        reg.register(ident)
        result = reg.resolve_seat("华泰益田", _utc("2025-01-01T00:00:00+00:00"))
        assert result.seat_id == "seat-001"

    def test_renamed_seat_resolves_historical(self):
        reg = SeatIdentityRegistry()
        old = SeatIdentity(
            seat_id="seat-001",
            canonical_name="老华泰营业部",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2010-01-01T00:00:00+00:00"),
            valid_to=_utc("2020-06-30T00:00:00+00:00"),
            aliases=("华泰旧名",),
        )
        new = SeatIdentity(
            seat_id="seat-001",
            canonical_name="新华泰营业部",
            seat_type=SeatType.BROKER_BRANCH,
            valid_from=_utc("2020-07-01T00:00:00+00:00"),
            valid_to=None,
            aliases=("华泰新名",),
        )
        reg.register(old)
        reg.register(new)

        result = reg.resolve_seat("华泰旧名", _utc("2015-01-01T00:00:00+00:00"))
        assert result.seat_id == "seat-001"
        assert result.canonical_name == "老华泰营业部"

        with pytest.raises(Exception):
            reg.resolve_seat("华泰旧名", _utc("2025-01-01T00:00:00+00:00"))

        result2 = reg.resolve_seat("华泰新名", _utc("2025-01-01T00:00:00+00:00"))
        assert result2.canonical_name == "新华泰营业部"

    def test_seat_id_is_string_not_numeric(self):
        reg = SeatIdentityRegistry()
        ident = SeatIdentity(
            seat_id="seat-abc-123",
            canonical_name="TestSeat",
            seat_type=SeatType.INSTITUTION,
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
        )
        reg.register(ident)
        result = reg.resolve_seat("TestSeat", _utc("2025-01-01T00:00:00+00:00"))
        assert isinstance(result.seat_id, str)
        # Must not be a pure numeric ID
        assert not result.seat_id.isdigit()

    def test_unresolved_returns_unknown(self):
        reg = SeatIdentityRegistry()
        result = reg.resolve_seat("未知席位名称", _utc("2025-01-01T00:00:00+00:00"))
        assert result.resolved is False
        assert result.seat_type == SeatType.UNKNOWN
        assert result.identity_confidence == 0.0
        # Unknown ID should be a stable string hash
        assert result.seat_id.startswith("UNKNOWN-")

    def test_seat_type_classification(self):
        """Seat types are enumerated, not free strings."""
        assert SeatType.BROKER_BRANCH.value == "BROKER_BRANCH"
        assert SeatType.INSTITUTION.value == "INSTITUTION"
        assert SeatType.NORTHBOUND_SH.value == "NORTHBOUND_SH"
        assert SeatType.NORTHBOUND_SZ.value == "NORTHBOUND_SZ"
        assert SeatType.SPECIAL_SEAT.value == "SPECIAL_SEAT"
        assert SeatType.UNKNOWN.value == "UNKNOWN"

    def test_identity_confidence_present(self):
        reg = SeatIdentityRegistry()
        ident = SeatIdentity(
            seat_id="seat-001",
            canonical_name="高置信席位",
            seat_type=SeatType.INSTITUTION,
            identity_confidence=0.95,
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
        )
        reg.register(ident)
        result = reg.resolve_seat("高置信席位", _utc("2025-01-01T00:00:00+00:00"))
        assert result.identity_confidence == 0.95
