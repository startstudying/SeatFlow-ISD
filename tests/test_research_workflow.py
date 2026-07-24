from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch
from zoneinfo import ZoneInfo

from seatflow import research_workflow
from seatflow.backtest import FailClosedBacktestError, PredictionSignal
from seatflow.data.contracts import (
    DailyBar,
    DragonTigerSummary,
    SuspensionRecord,
    SuspensionResponseEvidence,
)
from seatflow.research import ResearchSample
from seatflow.research_workflow import HistoricalResearchConfig, run_historical_research


class _HistoricalProvider:
    def __init__(self) -> None:
        self.calendar = tuple(_trading_days(date(2023, 1, 3), date(2026, 1, 16)))

    def fetch_trading_calendar(self, _: date, __: date) -> tuple[date, ...]:
        return self.calendar

    def fetch_dragon_tiger_summaries(self, trade_date: date) -> tuple[DragonTigerSummary, ...]:
        return (
            DragonTigerSummary(
                trade_date=trade_date,
                stock_id="000001.SZ",
                stock_name="fixture",
                buy_amount=Decimal("100"),
                sell_amount=Decimal("80"),
                net_amount=Decimal("20"),
                reason="fixture",
                source_name="tushare",
                source_version="top_list",
                available_time=datetime.combine(
                    trade_date, datetime.min.time(), ZoneInfo("Asia/Shanghai")
                ),
            ),
        )

    def fetch_daily_bar_history(
        self, stock_id: str, start_date: date, end_date: date
    ) -> tuple[DailyBar, ...]:
        return tuple(
            _daily_bar(trade_date, stock_id, index)
            for index, trade_date in enumerate(self.calendar)
            if start_date <= trade_date <= end_date
        )

    def fetch_index_close_history(
        self, _: str, start_date: date, end_date: date
    ) -> tuple[tuple[date, Decimal], ...]:
        return tuple(
            (trade_date, Decimal("4000") + Decimal(index))
            for index, trade_date in enumerate(self.calendar)
            if start_date <= trade_date <= end_date
        )

    def fetch_suspension_evidence(self, trade_date: date) -> SuspensionResponseEvidence:
        return _suspension_evidence(trade_date)


class _MissingExecutionBarProvider(_HistoricalProvider):
    def fetch_daily_bar_history(
        self, stock_id: str, start_date: date, end_date: date
    ) -> tuple[DailyBar, ...]:
        return tuple(
            bar
            for bar in super().fetch_daily_bar_history(stock_id, start_date, end_date)
            if bar.trade_date != date(2026, 1, 5)
        )


class HistoricalResearchWorkflowTest(unittest.TestCase):
    def test_rejects_future_blind_range_before_provider_collection(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "future blind labels"):
                run_historical_research(
                    HistoricalResearchConfig(
                        start_date=date(2023, 1, 1),
                        end_date=date(2026, 7, 20),
                        data_root=root / "data",
                        output_root=root / "artifacts",
                    ),
                    _HistoricalProvider(),  # type: ignore[arg-type]
                    None,
                )

    def test_runs_fixed_protocol_from_cached_capable_provider(self) -> None:
        provider = _HistoricalProvider()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = run_historical_research(
                HistoricalResearchConfig(
                    start_date=date(2023, 1, 1),
                    end_date=date(2026, 1, 20),
                    data_root=root / "data",
                    output_root=root / "artifacts",
                    seeds=(7,),
                    top_n=1,
                    akshare_sample_count=0,
                ),
                provider,  # type: ignore[arg-type]
                None,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            lineage_path_exists = Path(report["lineage"]["lineage_path"]).exists()

        self.assertGreater(result.sample_count, 500)
        self.assertEqual(result.effective_end_date, provider.calendar[-1])
        self.assertFalse(report["release_claims"]["model_valid"])
        self.assertFalse(report["release_claims"]["competition_ready"])
        self.assertEqual(report["seat_detail_scope"], "institutional_proxy")
        self.assertEqual(report["label_status"], "observed")
        self.assertEqual(report["blind_status"], "contaminated_oos")
        self.assertEqual(report["model_family"], "event_logistic_baseline_v1")
        self.assertEqual(report["evaluation_manifest"]["split_status"], "contaminated_oos")
        self.assertIn("protocol", report["run_hashes"])
        self.assertTrue(lineage_path_exists)
        self.assertEqual(report["backtest"]["mode"], "real")
        self.assertFalse(report["backtest"]["claims"]["real_backtest_valid"])
        self.assertTrue(report["backtest"]["claims"]["negative_result"])
        self.assertEqual(report["backtest"]["status"], "fail_closed")
        self.assertIn(
            "fill_rate",
            report["backtest"]["failure_reason"],
        )
        self.assertIn("orders", report["backtest"])
        self.assertIn("fills", report["backtest"])
        self.assertGreater(report["backtest"]["audit"]["execution"]["order_count"], 0)
        self.assertEqual(
            report["backtest"]["audit"]["real_execution_quality"]["passed"],
            False,
        )
        self.assertIn("execution", report["backtest"]["diagnostics"])
        self.assertEqual(
            report["release_claims"]["result_interpretation"],
            "negative_result_contaminated_oos_not_final_blind",
        )

    def test_retains_fail_closed_backtest_negative_result(self) -> None:
        provider = _MissingExecutionBarProvider()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = run_historical_research(
                HistoricalResearchConfig(
                    start_date=date(2023, 1, 1),
                    end_date=date(2026, 1, 20),
                    data_root=root / "data",
                    output_root=root / "artifacts",
                    seeds=(7,),
                    top_n=1,
                    akshare_sample_count=0,
                ),
                provider,  # type: ignore[arg-type]
                None,
            )
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            negative_path = Path(str(report["backtest"]["negative_result_path"]))
            negative_result_exists = negative_path.is_file()
            tradability_path = root / "artifacts" / "backtest" / "execution_tradability.jsonl"
            tradability_rows = [
                json.loads(line)
                for line in tradability_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(result.backtest_metrics["negative_result"])
        self.assertEqual(report["backtest"]["status"], "fail_closed")
        self.assertFalse(report["backtest"]["claims"]["real_backtest_valid"])
        self.assertTrue(negative_result_exists)
        self.assertEqual(tradability_rows[0]["status"], "unclassified_data_gap")
        self.assertEqual(tradability_rows[0]["suspension_daily_query_sha256"], "a" * 64)

    def test_execution_preflight_writes_sorted_cache_and_tail_gaps_before_engine(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = HistoricalResearchConfig(
                start_date=date(2026, 1, 2),
                end_date=date(2026, 1, 6),
                data_root=root / "data",
                output_root=root / "artifacts",
                seeds=(7,),
                top_n=1,
                akshare_sample_count=0,
            )
            samples = (
                _research_sample(date(2026, 1, 2), "000002.SZ"),
                _research_sample(date(2026, 1, 2), "000001.SZ"),
            )
            bars = (
                _daily_bar(date(2026, 1, 2), "000002.SZ", 0),
                _daily_bar(date(2026, 1, 2), "000001.SZ", 0),
                _daily_bar(date(2026, 1, 6), "000001.SZ", 1),
            )
            detail_path = config.output_root / "backtest" / "execution_tradability.jsonl"

            with patch("seatflow.research_workflow.BacktestEngine") as engine:
                with self.assertRaisesRegex(
                    FailClosedBacktestError, r"2 unclassified missing execution bar\(s\)"
                ) as error:
                    research_workflow._backtest_contaminated_oos(
                        config,
                        samples,
                        probabilities=(0.2, 0.8),
                        bars=bars,
                        benchmark_closes=(),
                        calendar=(date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)),
                        evaluation_manifest_sha256="fixture-manifest",
                        fetch_suspension_evidence=_suspension_evidence,
                    )
                engine.assert_not_called()
            self.assertIn(str(detail_path), str(error.exception))

            details = [
                json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            details,
            [
                {
                    "as_of_date": "2026-01-02",
                    "execution_date": "2026-01-05",
                    "gap_type": "cache_gap",
                    "retrieved_at": "2026-01-05T18:00:00+08:00",
                    "source": "tushare:suspend_d",
                    "status": "unclassified_data_gap",
                    "stock_id": "000001.SZ",
                    "suspend_timing": None,
                    "suspend_type": None,
                    "suspension_daily_query_sha256": "a" * 64,
                    "suspension_query_error": None,
                    "suspension_row_payload_sha256": None,
                },
                {
                    "as_of_date": "2026-01-02",
                    "execution_date": "2026-01-05",
                    "gap_type": "tail_truncation",
                    "retrieved_at": "2026-01-05T18:00:00+08:00",
                    "source": "tushare:suspend_d",
                    "status": "unclassified_data_gap",
                    "stock_id": "000002.SZ",
                    "suspend_timing": None,
                    "suspend_type": None,
                    "suspension_daily_query_sha256": "a" * 64,
                    "suspension_query_error": None,
                    "suspension_row_payload_sha256": None,
                },
            ],
        )

    def test_execution_preflight_classifies_confirmed_suspension_and_audits_calendar(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calendar = (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6))
            signals = (
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 2, 18, 0),
                ),
            )
            confirmed = research_workflow._require_execution_bar_completeness(
                output_root=root,
                signals=signals,
                bars=(),
                trading_calendar=calendar,
                fetch_suspension_evidence=lambda trade_date: _suspension_evidence(
                    trade_date,
                    ("000001.SZ",) if trade_date == calendar[1] else (),
                ),
            )
            detail_rows = [
                json.loads(line)
                for line in (root / "backtest" / "execution_tradability.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            query_rows = [
                json.loads(line)
                for line in (root / "backtest" / "suspension_query_evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(confirmed, {calendar[1]: ("000001.SZ",)})
        self.assertEqual(detail_rows[0]["status"], "confirmed_suspension_no_daily_bar")
        self.assertEqual(detail_rows[0]["suspension_row_payload_sha256"], "b" * 64)
        self.assertEqual(
            [row["trade_date"] for row in query_rows],
            [day.isoformat() for day in calendar],
        )
        self.assertTrue(all(row["daily_query_sha256"] == "a" * 64 for row in query_rows))

    def test_threshold_sensitivity_records_no_trade_policy_without_mutating_main_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = HistoricalResearchConfig(
                start_date=date(2026, 1, 2),
                end_date=date(2026, 1, 5),
                data_root=root / "data",
                output_root=root / "artifacts",
                top_n=1,
            )
            sensitivity = research_workflow._backtest_threshold_sensitivity(
                config,
                (_research_sample(date(2026, 1, 2), "000001.SZ"),),
                probabilities=(0.8,),
                bars=(
                    _daily_bar(date(2026, 1, 2), "000001.SZ", 0),
                    _daily_bar(date(2026, 1, 5), "000001.SZ", 1),
                ),
                benchmark_closes=(
                    (date(2026, 1, 2), Decimal("100")),
                    (date(2026, 1, 5), Decimal("100")),
                ),
                calendar=(date(2026, 1, 2), date(2026, 1, 5)),
                evaluation_manifest_sha256="fixture-manifest",
                fetch_suspension_evidence=_suspension_evidence,
            )

        threshold_rows = cast(list[dict[str, object]], sensitivity["thresholds"])
        rows = {str(row["policy"]): row for row in threshold_rows}
        self.assertEqual(rows["threshold_0.50"]["selected_signal_count"], 1)
        self.assertEqual(rows["no_trade_policy"]["selected_signal_count"], 0)
        self.assertEqual(rows["no_trade_policy"]["no_trade_days"], 1)
        self.assertIn("gross_return", rows["no_trade_policy"])
        self.assertIn("net_return", rows["no_trade_policy"])
        self.assertIn("turnover", rows["no_trade_policy"])
        self.assertIn("fill_quality", rows["no_trade_policy"])


def _trading_days(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return values


def _daily_bar(trade_date: date, stock_id: str, index: int) -> DailyBar:
    base = Decimal("10") + Decimal(index) * Decimal("0.01")
    return DailyBar(
        trade_date=trade_date,
        stock_id=stock_id,
        open_price=base,
        high_price=base + Decimal("0.20"),
        low_price=base - Decimal("0.10"),
        close_price=base + Decimal("0.10"),
        volume=Decimal("100000"),
        amount=Decimal("1000000"),
        source_name="tushare",
        source_version="tushare-pro-rest",
        available_time=datetime.combine(trade_date, datetime.min.time(), ZoneInfo("Asia/Shanghai")),
        pre_close=base - Decimal("0.01"),
        volume_unit="hand",
        amount_unit="thousand_cny",
    )


def _suspension_evidence(
    trade_date: date, stock_ids: tuple[str, ...] = ()
) -> SuspensionResponseEvidence:
    retrieved_at = datetime.combine(
        trade_date, datetime.min.time(), ZoneInfo("Asia/Shanghai")
    ).replace(hour=18)
    records = tuple(
        SuspensionRecord(
            stock_id=stock_id,
            trade_date=trade_date,
            suspend_type="S",
            suspend_timing="AM",
            source="tushare:suspend_d",
            retrieved_at=retrieved_at,
            row_payload_sha256="b" * 64,
        )
        for stock_id in stock_ids
    )
    return SuspensionResponseEvidence(
        trade_date=trade_date,
        source="tushare:suspend_d",
        retrieved_at=retrieved_at,
        source_row_count=len(records),
        daily_query_sha256="a" * 64,
        records=records,
    )


def _research_sample(as_of_date: date, stock_id: str) -> ResearchSample:
    return ResearchSample(
        sample_id=f"{as_of_date.isoformat()}:{stock_id}",
        as_of_date=as_of_date,
        stock_id=stock_id,
        feature_names=("fixture",),
        features=(0.0,),
        label_return=0.0,
        label=0,
        label_start_date=as_of_date + timedelta(days=1),
        label_end_date=as_of_date + timedelta(days=2),
    )
