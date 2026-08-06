"""Tests for reconciliation (Task 6)."""

from datetime import datetime


from seatflow.ingestion.models import BronzeRecord, SourceConfidence, compute_payload_hash
from seatflow.ingestion.reconcile import (
    reconcile_field,
    reconcile_records,
    RULE_VERSION,
)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _make_bronze(
    source: str = "tushare",
    record_id: str = "rec-001",
    payload: dict | None = None,
    confidence: SourceConfidence = SourceConfidence.HIGH,
) -> BronzeRecord:
    raw = payload or {"buy_amount": 1000000, "ts_code": "000001.SZ"}
    return BronzeRecord(
        event_time=_utc("2026-08-05T15:00:00+00:00"),
        available_time=_utc("2026-08-05T20:00:00+00:00"),
        ingested_at=_utc("2026-08-06T01:00:00+00:00"),
        source=source,
        source_record_id=record_id,
        source_version="v1",
        raw_payload=raw,
        raw_payload_hash=compute_payload_hash(raw),
        ingestion_run_id="run-001",
        source_confidence=confidence,
    )


class TestConflictPreservation:
    def test_conflicting_amounts_preserve_both_bronze(self):
        rec_a = _make_bronze("tushare", "r1", {"buy_amount": 1_000_000})
        rec_b = _make_bronze("akshare", "r2", {"buy_amount": 1_100_000}, SourceConfidence.MEDIUM)
        records = [rec_a, rec_b]
        value, report = reconcile_field(
            "key-001", "buy_amount", records,
            field_extractor=lambda p: p.get("buy_amount"),
        )
        assert report is not None
        assert len(report.sources) == 2
        # Both source values preserved
        source_names = {sv.source for sv in report.sources}
        assert source_names == {"tushare", "akshare"}

    def test_conflict_report_contains_source_values(self):
        rec_a = _make_bronze("tushare", "r1", {"buy_amount": 1_000_000})
        rec_b = _make_bronze("akshare", "r2", {"buy_amount": 1_100_000}, SourceConfidence.MEDIUM)
        _, report = reconcile_field(
            "key-001", "buy_amount", [rec_a, rec_b],
            field_extractor=lambda p: p.get("buy_amount"),
        )
        assert report is not None
        values = {sv.value for sv in report.sources}
        assert values == {1_000_000, 1_100_000}

    def test_selected_value_is_higher_confidence(self):
        rec_a = _make_bronze("tushare", "r1", {"buy_amount": 1_000_000}, SourceConfidence.HIGH)
        rec_b = _make_bronze("akshare", "r2", {"buy_amount": 1_100_000}, SourceConfidence.MEDIUM)
        value, report = reconcile_field(
            "key-001", "buy_amount", [rec_a, rec_b],
            field_extractor=lambda p: p.get("buy_amount"),
        )
        assert report is not None
        assert value == 1_000_000  # HIGH confidence selected
        assert report.selected_source == "tushare"

    def test_conflict_report_has_rule_version(self):
        rec_a = _make_bronze("tushare", "r1", {"x": 1})
        rec_b = _make_bronze("akshare", "r2", {"x": 2}, SourceConfidence.MEDIUM)
        _, report = reconcile_field(
            "key-001", "x", [rec_a, rec_b],
            field_extractor=lambda p: p.get("x"),
        )
        assert report is not None
        assert report.rule_version == RULE_VERSION

    def test_no_conflict_returns_none_report(self):
        rec_a = _make_bronze("tushare", "r1", {"buy_amount": 1_000_000})
        rec_b = _make_bronze("akshare", "r2", {"buy_amount": 1_000_000}, SourceConfidence.MEDIUM)
        value, report = reconcile_field(
            "key-001", "buy_amount", [rec_a, rec_b],
            field_extractor=lambda p: p.get("buy_amount"),
        )
        assert report is None
        assert value == 1_000_000


class TestReconcileRecords:
    def test_reconcile_multiple_fields(self):
        records_by_key = {
            "key-001": [
                _make_bronze("tushare", "r1", {"buy_amount": 1_000_000, "sell_amount": 500_000}),
                _make_bronze("akshare", "r2", {"buy_amount": 1_100_000, "sell_amount": 500_000}, SourceConfidence.MEDIUM),
            ],
        }
        result = reconcile_records(
            records_by_key,
            field_specs={
                "buy_amount": lambda p: p.get("buy_amount"),
                "sell_amount": lambda p: p.get("sell_amount"),
            },
        )
        rec = result["key-001"]
        assert rec.fields["buy_amount"] == 1_000_000  # HIGH confidence
        assert rec.fields["sell_amount"] == 500_000  # no conflict
        assert len(rec.conflict_reports) == 1  # only buy_amount conflicted
        assert rec.conflict_reports[0].field_name == "buy_amount"

    def test_reconciled_record_preserves_source_ids(self):
        records_by_key = {
            "key-001": [
                _make_bronze("tushare", "r1", {"x": 1}),
                _make_bronze("akshare", "r2", {"x": 1}),
            ],
        }
        result = reconcile_records(records_by_key, field_specs={"x": lambda p: p.get("x")})
        assert len(result["key-001"].source_record_ids) == 2
