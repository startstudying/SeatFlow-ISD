"""Tests for IngestionManifest and manifest hashing."""

from datetime import datetime, timezone

import pytest

from seatflow.ingestion.manifest import build_manifest
from seatflow.ingestion.models import (
    BronzeRecord,
    IngestionManifest,
    SourceConfidence,
    compute_payload_hash,
)


def _utc(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)


def _make_bronze(record_id: str = "rec-001", payload: dict | None = None) -> BronzeRecord:
    raw = payload or {"close": 10.5}
    return BronzeRecord(
        event_time=_utc("2026-08-05T15:00:00+00:00"),
        available_time=_utc("2026-08-05T20:00:00+00:00"),
        ingested_at=_utc("2026-08-06T01:00:00+00:00"),
        source="tushare",
        source_record_id=record_id,
        source_version="v1",
        raw_payload=raw,
        raw_payload_hash=compute_payload_hash(raw),
        ingestion_run_id="run-001",
        source_confidence=SourceConfidence.HIGH,
    )


class TestManifestDeterministicHash:
    def test_same_records_same_hash(self):
        records = [_make_bronze("r1"), _make_bronze("r2")]
        m1 = build_manifest(
            records,
            ingestion_run_id="run-001",
            source="tushare",
            source_version="v1",
        )
        m2 = build_manifest(
            records,
            ingestion_run_id="run-001",
            source="tushare",
            source_version="v1",
        )
        assert m1.manifest_hash == m2.manifest_hash

    def test_different_records_different_hash(self):
        records_a = [_make_bronze("r1", {"close": 10.5})]
        records_b = [_make_bronze("r1", {"close": 11.0})]
        m1 = build_manifest(records_a, ingestion_run_id="run-001", source="tushare", source_version="v1")
        m2 = build_manifest(records_b, ingestion_run_id="run-001", source="tushare", source_version="v1")
        assert m1.manifest_hash != m2.manifest_hash

    def test_record_count(self):
        records = [_make_bronze(f"r{i}") for i in range(5)]
        m = build_manifest(records, ingestion_run_id="run-001", source="tushare", source_version="v1")
        assert m.record_count == 5
        assert len(m.record_hashes) == 5

    def test_manifest_hash_non_empty(self):
        records = [_make_bronze()]
        m = build_manifest(records, ingestion_run_id="run-001", source="tushare", source_version="v1")
        assert m.manifest_hash != ""
        assert len(m.manifest_hash) == 64  # SHA-256 hex

    def test_empty_manifest(self):
        m = build_manifest([], ingestion_run_id="run-empty", source="tushare", source_version="v1")
        assert m.record_count == 0
        assert m.record_hashes == []
        assert m.manifest_hash != ""

    def test_hash_order_invariant(self):
        r1 = _make_bronze("r1")
        r2 = _make_bronze("r2")
        m1 = build_manifest([r1, r2], ingestion_run_id="run-001", source="tushare", source_version="v1")
        m2 = build_manifest([r2, r1], ingestion_run_id="run-001", source="tushare", source_version="v1")
        # Hashes are sorted, so order doesn't matter
        assert m1.manifest_hash == m2.manifest_hash
