"""Tests for Bronze record models and contracts."""

from datetime import datetime, timezone

import pytest

from seatflow.ingestion.models import (
    AvailabilityPolicy,
    BronzeRecord,
    SourceConfidence,
    compute_payload_hash,
)
from seatflow.domain.time import validate_available_after_event


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)


def _aware() -> datetime:
    return datetime(2026, 8, 6, 15, 0, 0, tzinfo=timezone.utc)


def _make_bronze(**overrides) -> BronzeRecord:
    defaults = dict(
        event_time=_utc("2026-08-05T15:00:00+00:00"),
        available_time=_utc("2026-08-05T20:00:00+00:00"),
        ingested_at=_aware(),
        source="tushare",
        source_record_id="rec-001",
        source_version="v1",
        revision_no=0,
        raw_payload={"close": 10.5},
        raw_payload_hash="deadbeef",
        ingestion_run_id="run-001",
        source_confidence=SourceConfidence.HIGH,
    )
    defaults.update(overrides)
    return BronzeRecord(**defaults)


# ── BronzeRecord mandatory fields ──────────────────────────────────────


class TestBronzeRecordMandatory:
    def test_all_mandatory_fields_present(self):
        rec = _make_bronze()
        assert rec.event_time is not None
        assert rec.available_time is not None
        assert rec.ingested_at is not None
        assert rec.source == "tushare"
        assert rec.source_record_id == "rec-001"
        assert rec.source_version == "v1"
        assert rec.revision_no == 0
        assert rec.raw_payload_hash == "deadbeef"
        assert rec.ingestion_run_id == "run-001"
        assert rec.source_confidence == SourceConfidence.HIGH

    def test_missing_event_time_raises(self):
        with pytest.raises(Exception):
            _make_bronze(event_time=None)

    def test_missing_available_time_raises(self):
        with pytest.raises(Exception):
            _make_bronze(available_time=None)

    def test_missing_source_raises(self):
        with pytest.raises(Exception):
            _make_bronze(source=None)

    def test_missing_source_record_id_raises(self):
        with pytest.raises(Exception):
            _make_bronze(source_record_id=None)

    def test_missing_ingestion_run_id_raises(self):
        with pytest.raises(Exception):
            _make_bronze(ingestion_run_id=None)


# ── Timezone awareness ─────────────────────────────────────────────────


class TestTimezoneAwareness:
    def test_naive_event_time_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _make_bronze(event_time=datetime(2026, 8, 5))

    def test_naive_available_time_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _make_bronze(available_time=datetime(2026, 8, 5))

    def test_naive_ingested_at_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _make_bronze(ingested_at=datetime(2026, 8, 6))

    def test_aware_timestamps_accepted(self):
        rec = _make_bronze()
        assert rec.event_time.tzinfo is not None
        assert rec.available_time.tzinfo is not None


# ── Availability ordering ──────────────────────────────────────────────


class TestAvailabilityOrdering:
    def test_available_after_event_ok(self):
        validate_available_after_event(
            _utc("2026-08-05T15:00:00+00:00"),
            _utc("2026-08-05T20:00:00+00:00"),
        )

    def test_available_before_event_rejected(self):
        with pytest.raises(ValueError, match="available_time"):
            validate_available_after_event(
                _utc("2026-08-05T20:00:00+00:00"),
                _utc("2026-08-05T15:00:00+00:00"),
            )

    def test_pre_announced_allowed_when_policy_permits(self):
        validate_available_after_event(
            _utc("2026-08-05T20:00:00+00:00"),
            _utc("2026-08-05T15:00:00+00:00"),
            allow_pre_announced=True,
        )

    def test_availability_policy_enforced(self):
        policy = AvailabilityPolicy(allow_pre_announced=False)
        with pytest.raises(ValueError):
            policy.validate(
                _utc("2026-08-05T20:00:00+00:00"),
                _utc("2026-08-05T15:00:00+00:00"),
            )


# ── Payload hash ───────────────────────────────────────────────────────


class TestPayloadHash:
    def test_deterministic(self):
        payload = {"close": 10.5, "volume": 1000}
        h1 = compute_payload_hash(payload)
        h2 = compute_payload_hash(payload)
        assert h1 == h2

    def test_different_payload_different_hash(self):
        h1 = compute_payload_hash({"close": 10.5})
        h2 = compute_payload_hash({"close": 10.6})
        assert h1 != h2

    def test_key_order_invariant(self):
        h1 = compute_payload_hash({"a": 1, "b": 2})
        h2 = compute_payload_hash({"b": 2, "a": 1})
        assert h1 == h2


# ── Immutability ───────────────────────────────────────────────────────


class TestBronzeImmutability:
    def test_frozen_model(self):
        rec = _make_bronze()
        with pytest.raises(Exception):
            rec.source = "akshare"
