import json
import unittest
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.cli import main
from seatflow.data.audit import DataAuditPolicy, audit_market_data, write_audit_report
from seatflow.data.providers import ProviderCapabilityError, ProviderStatus, SampleProvider
from seatflow.data.records import DragonTigerRecord


class DataAuditTest(unittest.TestCase):
    def test_synthetic_provider_stays_fail_closed(self) -> None:
        report = audit_market_data(SampleProvider(), [date(2026, 1, 15)], require_seat_detail=True)

        self.assertFalse(report.data_readiness)
        self.assertFalse(report.full_seat_ready)
        self.assertEqual(report.seat_detail_scope, "synthetic")
        self.assertEqual(report.data_scope, "synthetic")
        self.assertIn("synthetic provider data", " ".join(report.blocked_reasons))

    def test_live_complete_scope_requires_successful_endpoints(self) -> None:
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

        report = audit_market_data(
            LicensedSeatProvider(),
            [date(2026, 1, 15)],
            require_seat_detail=True,
            policy=DataAuditPolicy(minimum_trade_days=1),
        )

        self.assertTrue(report.data_readiness)
        self.assertTrue(report.full_seat_ready)
        self.assertEqual(report.seat_detail_scope, "complete_seat")
        self.assertEqual(report.data_scope, "real")

    def test_missing_seat_endpoint_is_a_blocker_without_scope_fallback(self) -> None:
        class SummaryOnlyProvider(SampleProvider):
            def fetch_dragon_tiger_seat_events(
                self, trade_date: date
            ) -> tuple[DragonTigerRecord, ...]:
                raise ProviderCapabilityError("seat token=private-value is unavailable")

        report = audit_market_data(
            SummaryOnlyProvider(), [date(2026, 1, 15)], require_seat_detail=True
        )

        self.assertFalse(report.data_readiness)
        self.assertEqual(report.seat_detail_scope, "none")
        if report.seat_detail is None:
            self.fail("seat-detail audit should exist when it is required")
        self.assertNotIn("private-value", " ".join(report.seat_detail.errors))

    def test_report_is_written_as_json(self) -> None:
        report = audit_market_data(SampleProvider(), [date(2026, 1, 15)], require_seat_detail=False)
        with TemporaryDirectory() as temporary_directory:
            path = write_audit_report(Path(temporary_directory) / "audit.json", report)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["seat_detail_scope"], "none")
        self.assertFalse(loaded["data_readiness"])

    def test_cli_requires_a_verified_calendar_and_writes_fail_closed_sample_report(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = root / "calendar.json"
            calendar.write_text('["2026-01-15"]', encoding="utf-8")
            output = root / "audit.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "audit-data",
                        "--provider",
                        "sample",
                        "--start",
                        "2026-01-15",
                        "--end",
                        "2026-01-15",
                        "--trading-calendar-file",
                        str(calendar),
                        "--output",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertIn("report_path", stdout.getvalue())
        self.assertFalse(report["data_readiness"])
        self.assertEqual(report["seat_detail_scope"], "synthetic")
