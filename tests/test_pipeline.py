import json
import unittest
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.cli import main
from seatflow.data.contracts import DailyBar
from seatflow.data.providers import (
    ProviderCapabilityError,
    ProviderResponseError,
    ProviderStatus,
    SampleProvider,
)
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.pipeline import DataIngestionService


class DataIngestionPipelineTest(unittest.TestCase):
    def test_sample_ingestion_writes_versioned_raw_processed_features_and_graph(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = DataIngestionService(SampleProvider(), root)

            result = service.ingest_market_day(date(2026, 1, 15))
            feature_artifacts = service.build_features()
            graph_artifact = service.build_graph_snapshot(
                as_of_date=date(2026, 1, 15), decay_lambda=0.1, max_edge_age_days=20
            )

            self.assertTrue(all(report.passed for report in result.reports))
            self.assertEqual(len(result.artifacts), 6)
            self.assertEqual(len(feature_artifacts), 1)
            self.assertIsNotNone(graph_artifact)
            self.assertTrue((root / "raw" / "sample").exists())
            if graph_artifact is None:
                self.fail("sample graph artifact should exist")
            self.assertTrue(graph_artifact.manifest_path.exists())

    def test_historical_graph_build_ignores_later_snapshot_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            service = DataIngestionService(SampleProvider(), Path(temporary_directory))
            service.ingest_market_day(date(2026, 1, 15))
            service.ingest_market_day(date(2026, 1, 16))

            graph_artifact = service.build_graph_snapshot(
                as_of_date=date(2026, 1, 15), decay_lambda=0.1, max_edge_age_days=20
            )

            self.assertIsNotNone(graph_artifact)

    def test_required_seat_detail_fails_closed(self) -> None:
        class SummaryOnlyProvider(SampleProvider):
            name = "summary-only"

            def fetch_dragon_tiger_seat_events(
                self, trade_date: date
            ) -> tuple[DragonTigerRecord, ...]:
                raise ProviderCapabilityError("seat detail unavailable")

        with TemporaryDirectory() as temporary_directory:
            service = DataIngestionService(SummaryOnlyProvider(), Path(temporary_directory))

            with self.assertRaisesRegex(ProviderCapabilityError, "seat detail unavailable"):
                service.ingest_market_day(date(2026, 1, 15), require_seat_detail=True)

    def test_serializes_and_reloads_canonical_seat_provenance(self) -> None:
        class ProvenanceProvider(SampleProvider):
            name = "provenance"

            def fetch_dragon_tiger_seat_events(
                self, trade_date: date
            ) -> tuple[DragonTigerRecord, ...]:
                return (
                    DragonTigerRecord(
                        trade_date=trade_date,
                        seat_id="institutional-flow:2026-01-15:000001.SZ:value%3Areason:buy",
                        stock_id="000001.SZ",
                        side=TradeSide.BUY,
                        amount=100.0,
                        event_reason="reason",
                        source_name=self.name,
                        source_version="top_inst",
                        source_row_count=3,
                        source_side_set=("sell", "buy", "sell"),
                        identity_scope="institutional_flow_proxy",
                    ),
                )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = DataIngestionService(ProvenanceProvider(), root)
            service.ingest_market_day(date(2026, 1, 15))
            serialized = json.loads(
                (
                    root
                    / "raw"
                    / "provenance"
                    / "dragon_tiger_seat_event"
                    / "date=2026-01-15"
                    / "data.json"
                ).read_text(encoding="utf-8")
            )
            graph = service.build_graph_snapshot(
                as_of_date=date(2026, 1, 15), decay_lambda=0.1, max_edge_age_days=20
            )

        self.assertEqual(serialized[0]["source_row_count"], 3)
        self.assertEqual(serialized[0]["source_side_set"], ["buy", "sell"])
        self.assertEqual(serialized[0]["identity_scope"], "institutional_flow_proxy")
        self.assertIsNotNone(graph)

    def test_duplicate_canonical_seat_event_fails_before_snapshot_write(self) -> None:
        class DuplicateRecordProvider(SampleProvider):
            name = "duplicate-record"

            def fetch_dragon_tiger_seat_events(
                self, trade_date: date
            ) -> tuple[DragonTigerRecord, ...]:
                record = DragonTigerRecord(
                    trade_date=trade_date,
                    seat_id="named-seat",
                    stock_id="000001.SZ",
                    side=TradeSide.BUY,
                    amount=100.0,
                    event_reason="reason",
                    source_name=self.name,
                )
                return (record, record)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = DataIngestionService(DuplicateRecordProvider(), root)

            with self.assertRaisesRegex(ValueError, "duplicate canonical seat event"):
                service.ingest_market_day(date(2026, 1, 15))

            self.assertFalse(
                (
                    root
                    / "raw"
                    / "duplicate-record"
                    / "dragon_tiger_seat_event"
                    / "date=2026-01-15"
                    / "data.json"
                ).exists()
            )

    def test_graph_promotion_requires_a_matching_live_audit(self) -> None:
        class LicensedSeatProvider(SampleProvider):
            name = "licensed-seat"

            def status(self) -> ProviderStatus:
                return ProviderStatus(
                    provider=self.name,
                    configured=True,
                    supports_daily_bars=True,
                    supports_dragon_tiger_summary=True,
                    supports_seat_detail=True,
                    seat_detail_scope="complete_seat",
                    notes=(),
                )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = DataIngestionService(LicensedSeatProvider(), root)
            service.ingest_market_day(date(2026, 1, 15))
            graph = service.build_graph_snapshot(
                as_of_date=date(2026, 1, 15), decay_lambda=0.1, max_edge_age_days=20
            )
            self.assertIsNotNone(graph)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "provider_status": {"provider": "licensed-seat"},
                        "data_readiness": True,
                        "data_scope": "real",
                        "seat_detail_scope": "complete_seat",
                    }
                ),
                encoding="utf-8",
            )

            promoted = service.promote_graph_snapshot(
                as_of_date=date(2026, 1, 15),
                audit_report_path=audit,
                require_complete_seat=True,
            )
            manifest = json.loads(promoted.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["metadata"]["data_scope"], "real")
        self.assertEqual(manifest["metadata"]["seat_detail_scope"], "complete_seat")

    def test_synthetic_provider_graph_cannot_be_promoted(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            service = DataIngestionService(SampleProvider(), root)
            service.ingest_market_day(date(2026, 1, 15))
            service.build_graph_snapshot(
                as_of_date=date(2026, 1, 15), decay_lambda=0.1, max_edge_age_days=20
            )
            audit = root / "audit.json"
            audit.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "synthetic providers"):
                service.promote_graph_snapshot(
                    as_of_date=date(2026, 1, 15), audit_report_path=audit
                )

    def test_historical_backfill_skips_verified_snapshots_by_hash(self) -> None:
        class CountingProvider(SampleProvider):
            name = "counting"

            def __init__(self) -> None:
                super().__init__()
                self.daily_calls = 0

            def fetch_daily_bars(
                self, trade_date: date, stock_id: str | None = None
            ) -> tuple[DailyBar, ...]:
                self.daily_calls += 1
                return super().fetch_daily_bars(trade_date, stock_id)

        with TemporaryDirectory() as temporary_directory:
            provider = CountingProvider()
            service = DataIngestionService(provider, Path(temporary_directory))

            first = service.backfill_history([date(2026, 1, 15)])
            second = service.backfill_history([date(2026, 1, 15)])

        self.assertTrue(first.passed)
        self.assertTrue(second.passed)
        self.assertEqual(provider.daily_calls, 1)
        self.assertEqual(second.days[0].status, "skipped")
        self.assertIn("raw/daily_bar", second.days[0].hash_summary)

    def test_historical_backfill_records_failure_and_continues_fail_closed(self) -> None:
        class OneDayFailureProvider(SampleProvider):
            name = "one-day-failure"

            def fetch_daily_bars(
                self, trade_date: date, stock_id: str | None = None
            ) -> tuple[DailyBar, ...]:
                if trade_date == date(2026, 1, 16):
                    raise ProviderResponseError("provider token=private-value failed")
                return super().fetch_daily_bars(trade_date, stock_id)

        with TemporaryDirectory() as temporary_directory:
            service = DataIngestionService(OneDayFailureProvider(), Path(temporary_directory))

            result = service.backfill_history(
                [date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 19)],
                continue_on_error=True,
            )

        self.assertFalse(result.passed)
        self.assertFalse(result.stopped_early)
        self.assertEqual([day.status for day in result.days], ["ingested", "failed", "ingested"])
        self.assertEqual(result.days[1].error_type, "ProviderResponseError")
        self.assertNotIn("private-value", result.days[1].error_message or "")

    def test_historical_backfill_stops_early_without_continue_on_error(self) -> None:
        class OneDayFailureProvider(SampleProvider):
            name = "one-day-stop"

            def fetch_daily_bars(
                self, trade_date: date, stock_id: str | None = None
            ) -> tuple[DailyBar, ...]:
                if trade_date == date(2026, 1, 16):
                    raise ProviderResponseError("provider failed")
                return super().fetch_daily_bars(trade_date, stock_id)

        with TemporaryDirectory() as temporary_directory:
            service = DataIngestionService(OneDayFailureProvider(), Path(temporary_directory))

            result = service.backfill_history(
                [date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 19)]
            )

        self.assertFalse(result.passed)
        self.assertTrue(result.stopped_early)
        self.assertEqual([day.status for day in result.days], ["ingested", "failed"])

    def test_cli_backfill_history_reads_calendar_json_and_writes_summary(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = root / "calendar.json"
            output = root / "backfill.json"
            summary = root / "summary.json"
            calendar.write_text('{"trade_dates": ["2026-01-15"]}', encoding="utf-8")
            stdout = StringIO()

            with redirect_stdout(stdout):
                main(
                    [
                        "backfill-history",
                        "--provider",
                        "sample",
                        "--trading-calendar-file",
                        str(calendar),
                        "--data-root",
                        str(root / "data"),
                        "--output",
                        str(output),
                        "--summary",
                        str(summary),
                    ]
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            compact = json.loads(summary.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())

        self.assertTrue(report["passed"])
        self.assertEqual(report["succeeded"], 1)
        self.assertEqual(compact["days"][0]["status"], "ingested")
        self.assertEqual(printed["report_path"], str(output))
