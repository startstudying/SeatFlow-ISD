from __future__ import annotations

import gzip
import json
import unittest
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from seatflow.cli import build_parser
from seatflow.data.contracts import DailyBar
from seatflow.data.providers import ProviderError
from seatflow.research_cache_repair import (
    repair_research_cache,
    write_research_cache_repair_report,
)


class _HistoryProvider:
    name = "tushare"

    def __init__(self, rows: tuple[DailyBar, ...], error: str | None = None) -> None:
        self.rows = rows
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_daily_bar_history(
        self, stock_id: str, start_date: date, end_date: date
    ) -> tuple[DailyBar, ...]:
        self.calls.append((stock_id, start_date, end_date))
        if self.error is not None:
            raise ProviderError(self.error)
        return self.rows


class ResearchCacheRepairTest(unittest.TestCase):
    def test_merges_full_range_creates_backup_and_deduplicates_requests(self) -> None:
        start_date = date(2026, 1, 2)
        end_date = date(2026, 1, 5)
        first_date = date(2026, 1, 2)
        execution_date = date(2026, 1, 5)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_path = root / "custom" / "000001.SZ.json.gz"
            _write_cache(cache_path, (_daily_bar(first_date, "10"),))
            original_bytes = cache_path.read_bytes()
            missing_path = root / "missing.jsonl"
            _write_jsonl(
                missing_path,
                (
                    {
                        "stock_id": "000001.SZ",
                        "execution_date": execution_date.isoformat(),
                        "cache_path": str(cache_path),
                    },
                    {
                        "stock_id": "000001.SZ",
                        "execution_date": execution_date.isoformat(),
                        "cache_path": str(cache_path),
                    },
                ),
            )
            provider = _HistoryProvider(
                (_daily_bar(first_date, "11"), _daily_bar(execution_date, "12"))
            )

            report = repair_research_cache(
                provider, root, start_date, end_date, missing_path
            )

            result = report.results[0]
            self.assertTrue(report.passed)
            self.assertEqual(report.input_record_count, 2)
            self.assertEqual(report.to_dict()["requested_execution_bar_count"], 1)
            self.assertEqual(provider.calls, [("000001.SZ", start_date, end_date)])
            self.assertEqual(result.added, 1)
            self.assertEqual(result.replaced, 1)
            self.assertEqual(result.original_sha256, sha256(original_bytes).hexdigest())
            self.assertIsNotNone(result.backup_path)
            assert result.backup_path is not None
            self.assertEqual(result.backup_path.read_bytes(), original_bytes)
            self.assertEqual(result.backup_sha256, sha256(original_bytes).hexdigest())
            repaired_rows = _read_cache(cache_path)
            self.assertEqual(repaired_rows[0]["close_price"], "11")
            self.assertEqual(repaired_rows[1]["trade_date"], execution_date.isoformat())

    def test_unresolved_execution_bar_writes_negative_report(self) -> None:
        start_date = date(2026, 1, 2)
        end_date = date(2026, 1, 5)
        execution_date = date(2026, 1, 5)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_path = root / "missing.jsonl"
            _write_jsonl(
                missing_path,
                ({"stock_id": "000001.SZ", "execution_date": execution_date.isoformat()},),
            )
            report = repair_research_cache(
                _HistoryProvider((_daily_bar(start_date, "10"),)),
                root,
                start_date,
                end_date,
                missing_path,
            )
            output = root / "repair-report.json"
            write_research_cache_repair_report(output, report)

            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report.passed)
            self.assertEqual(
                report.results[0].unresolved_execution_dates, (execution_date,)
            )
            self.assertFalse(persisted["passed"])
            self.assertEqual(
                persisted["unresolved"],
                [{"stock_id": "000001.SZ", "execution_date": execution_date.isoformat()}],
            )

    def test_provider_error_is_reported_without_changing_existing_cache(self) -> None:
        start_date = date(2026, 1, 2)
        end_date = date(2026, 1, 5)
        execution_date = date(2026, 1, 5)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_path = (
                root
                / "raw"
                / "tushare"
                / "daily"
                / "stock=000001.SZ.start=2026-01-02.end=2026-01-05.json.gz"
            )
            _write_cache(cache_path, (_daily_bar(start_date, "10"),))
            original_bytes = cache_path.read_bytes()
            missing_path = root / "missing.jsonl"
            _write_jsonl(
                missing_path,
                ({"stock_id": "000001.SZ", "execution_date": execution_date.isoformat()},),
            )

            report = repair_research_cache(
                _HistoryProvider((), error="Tushare daily endpoint unavailable"),
                root,
                start_date,
                end_date,
                missing_path,
            )

            result = report.results[0]
            self.assertFalse(report.passed)
            self.assertEqual(result.error, "Tushare daily endpoint unavailable")
            self.assertEqual(result.unresolved_execution_dates, (execution_date,))
            self.assertEqual(cache_path.read_bytes(), original_bytes)
            self.assertIsNone(result.backup_path)

    def test_cli_exposes_required_cache_repair_parameters(self) -> None:
        args = build_parser().parse_args(
            (
                "repair-research-cache",
                "--provider",
                "tushare",
                "--data-root",
                "data/research/test",
                "--start",
                "2026-01-02",
                "--end",
                "2026-01-05",
                "--missing-execution-jsonl",
                "artifacts/missing.jsonl",
                "--output",
                "artifacts/repair-report.json",
            )
        )

        self.assertEqual(args.command, "repair-research-cache")
        self.assertEqual(args.provider, "tushare")
        self.assertEqual(args.start, date(2026, 1, 2))
        self.assertEqual(args.end, date(2026, 1, 5))
        self.assertEqual(args.data_root, Path("data/research/test"))


def _daily_bar(trade_date: date, close: str) -> DailyBar:
    close_price = Decimal(close)
    return DailyBar(
        trade_date=trade_date,
        stock_id="000001.SZ",
        open_price=close_price,
        high_price=close_price,
        low_price=close_price,
        close_price=close_price,
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="tushare",
        source_version="fixture",
        available_time=datetime.combine(trade_date, datetime.min.time(), ZoneInfo("Asia/Shanghai")),
        pre_close=close_price,
        volume_unit="hand",
        amount_unit="thousand_cny",
    )


def _write_cache(path: Path, bars: tuple[DailyBar, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"rows": [bar.to_dict() for bar in bars]}, stream)


def _read_cache(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload["rows"]


def _write_jsonl(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
