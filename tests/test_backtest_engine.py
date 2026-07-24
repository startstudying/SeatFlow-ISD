from __future__ import annotations

import unittest
from datetime import date, datetime, time
from decimal import Decimal
from inspect import signature
from typing import TypedDict, cast
from unittest.mock import patch

from seatflow.backtest import (
    BacktestConfig,
    BacktestEngine,
    ExecutionOrder,
    FailClosedBacktestError,
    OosFold,
    PredictionSignal,
    TradeFill,
)
from seatflow.data.contracts import DailyBar


def _bar(
    day: date,
    stock_id: str,
    open_price: str,
    close_price: str,
    volume: str = "100000",
    pre_close: str = "10",
) -> DailyBar:
    open_decimal = Decimal(open_price)
    close_decimal = Decimal(close_price)
    high = max(open_decimal, close_decimal)
    low = min(open_decimal, close_decimal)
    return DailyBar(
        trade_date=day,
        stock_id=stock_id,
        open_price=open_decimal,
        high_price=high,
        low_price=low,
        close_price=close_decimal,
        volume=Decimal(volume),
        amount=Decimal(volume) * close_decimal,
        source_name="fixture",
        available_time=datetime.combine(day, time(18, 0)),
        pre_close=Decimal(pre_close),
        source_version="fixture-v1",
        volume_unit="shares",
        amount_unit="yuan",
    )


class _PoisonExecutionBar:
    """Fails if order construction inspects any t+1 market field."""

    def __init__(self, day: date, stock_id: str) -> None:
        self.trade_date = day
        self.stock_id = stock_id
        self.market_field_reads: list[str] = []

    def __getattr__(self, name: str) -> Decimal:
        self.market_field_reads.append(name)
        raise AssertionError(f"order construction read execution bar field: {name}")


class _RunInputs(TypedDict):
    signals: tuple[PredictionSignal, ...]
    trading_calendar: tuple[date, ...]
    universe_by_date: dict[date, tuple[str, ...]]
    benchmark_returns: dict[date, Decimal]


class BacktestEngineTest(unittest.TestCase):
    def test_minimum_score_abstains_when_all_scores_below_threshold(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))

        report = BacktestEngine(
            BacktestConfig(
                run_id="threshold-abstain",
                top_n=2,
                minimum_score=Decimal("0.60"),
                initial_cash=Decimal("10000"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("0.59"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000002.SZ",
                    score=Decimal("0.50"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(
                _bar(calendar[1], "000001.SZ", "10", "10"),
                _bar(calendar[1], "000002.SZ", "10", "10"),
            ),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ", "000002.SZ")},
            benchmark_returns={calendar[1]: Decimal("0")},
        )

        self.assertEqual(report.orders, ())
        selection = cast(dict[str, object], report.audit["selection"])
        self.assertEqual(selection["threshold_mode"], "abstain_below_minimum_score")
        self.assertEqual(selection["minimum_score"], "0.60")
        self.assertEqual(selection["selected_signal_count"], 0)
        self.assertEqual(selection["abstention_day_count"], 1)
        self.assertEqual(selection["exposure_day_count"], 0)

    def test_minimum_score_selects_exact_threshold_and_fewer_than_top_n(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))

        report = BacktestEngine(
            BacktestConfig(
                run_id="threshold-exact",
                top_n=3,
                minimum_score=Decimal("0.60"),
                initial_cash=Decimal("10000"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("0.60"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000002.SZ",
                    score=Decimal("0.59"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(
                _bar(calendar[1], "000001.SZ", "10", "10"),
                _bar(calendar[1], "000002.SZ", "10", "10"),
            ),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ", "000002.SZ")},
            benchmark_returns={calendar[1]: Decimal("0")},
        )

        self.assertEqual([order.stock_id for order in report.orders], ["000001.SZ"])
        selection = cast(dict[str, object], report.audit["selection"])
        self.assertEqual(selection["selected_signal_count"], 1)
        self.assertEqual(selection["abstention_day_count"], 0)
        self.assertEqual(selection["exposure_day_count"], 1)

    def test_minimum_score_counts_no_candidate_day_as_abstention(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))

        report = BacktestEngine(
            BacktestConfig(
                run_id="threshold-no-candidate",
                top_n=1,
                minimum_score=Decimal("0"),
                exclude_as_of_zero_volume_signals=True,
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(
                _bar(calendar[0], "000001.SZ", "10", "10", volume="0"),
                _bar(calendar[1], "000001.SZ", "10", "10"),
            ),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ",)},
            benchmark_returns={calendar[1]: Decimal("0")},
        )

        self.assertEqual(report.orders, ())
        selection = cast(dict[str, object], report.audit["selection"])
        self.assertEqual(selection["abstention_day_count"], 1)
        day_summaries = cast(list[dict[str, object]], selection["day_summaries"])
        self.assertEqual(day_summaries[0]["eligible_signal_count"], 0)
        self.assertTrue(day_summaries[0]["abstained"])

    def test_legacy_selection_keeps_top_n_without_threshold(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))

        report = BacktestEngine(
            BacktestConfig(
                run_id="legacy-threshold",
                top_n=1,
                initial_cash=Decimal("10000"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("-1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "000001.SZ", "10", "10"),),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ",)},
            benchmark_returns={calendar[1]: Decimal("0")},
        )

        self.assertEqual([order.stock_id for order in report.orders], ["000001.SZ"])
        selection = cast(dict[str, object], report.audit["selection"])
        self.assertEqual(selection["threshold_mode"], "legacy_top_n")
        self.assertIsNone(selection["minimum_score"])

    def test_fixture_smoke_uses_next_trading_day_lots_and_worst_oos(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 5))
        bars = (
            _bar(calendar[1], "000001.SZ", "10", "11"),
            _bar(calendar[2], "000001.SZ", "11", "12"),
        )
        report = BacktestEngine(
            BacktestConfig(
                run_id="fixture",
                top_n=1,
                initial_cash=Decimal("10000"),
                slippage_bps=Decimal("0"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=bars,
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ",)},
            benchmark_returns={calendar[1]: Decimal("0.01"), calendar[2]: Decimal("0.01")},
            oos_folds=(OosFold("fold-a", calendar[0], calendar[2]),),
        )

        self.assertFalse(report.claims["real_backtest_valid"])
        self.assertEqual(report.orders[0].execution_date, calendar[1])
        self.assertEqual(report.fills[0].quantity % 100, 0)
        self.assertEqual(report.metrics["worst_oos_fold"], "fold-a")

    def test_order_construction_cannot_inspect_execution_bar(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))
        poison = _PoisonExecutionBar(calendar[1], "000001.SZ")
        engine = BacktestEngine(
            BacktestConfig(
                run_id="no-next-day-read",
                top_n=1,
                initial_cash=Decimal("10000"),
                commission_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            )
        )

        def block_execution(order: ExecutionOrder, *_: object) -> TradeFill:
            return TradeFill(
                order=order,
                price=Decimal("0"),
                quantity=0,
                commission=Decimal("0"),
                stamp_tax=Decimal("0"),
                blocked=True,
                blocked_reason="test_execution_stub",
            )

        with patch.object(engine, "_execute", side_effect=block_execution) as execute:
            report = engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=calendar[0],
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(poison,),  # type: ignore[arg-type]
                trading_calendar=calendar,
                universe_by_date={calendar[0]: ("000001.SZ",)},
                benchmark_returns={calendar[1]: Decimal("0")},
            )

        self.assertNotIn("bar_map", signature(engine._build_orders).parameters)
        self.assertEqual(poison.market_field_reads, [])
        self.assertEqual(report.orders[0].target_notional, Decimal("10000"))
        execute.assert_called_once()

    def test_as_of_tradability_screen_excludes_known_limit_up_without_execution_bar(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))
        engine = BacktestEngine(
            BacktestConfig(
                run_id="as-of-screen",
                top_n=2,
                exclude_as_of_limit_up_signals=True,
                exclude_as_of_zero_volume_signals=True,
                min_as_of_close_price=Decimal("5"),
                excluded_stock_exchanges=("BJ",),
            )
        )

        report = engine.run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="920001.BJ",
                    score=Decimal("4"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000003.SZ",
                    score=Decimal("3"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("2"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000002.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(
                _bar(calendar[0], "920001.BJ", "10", "10", pre_close="10"),
                _bar(calendar[0], "000003.SZ", "4", "4", pre_close="4"),
                _bar(calendar[0], "000001.SZ", "11", "11", pre_close="10"),
                _bar(calendar[0], "000002.SZ", "10", "10", pre_close="10"),
                _bar(calendar[1], "920001.BJ", "10", "10", pre_close="10"),
                _bar(calendar[1], "000003.SZ", "4", "4", pre_close="4"),
                _bar(calendar[1], "000001.SZ", "11", "11", pre_close="11"),
                _bar(calendar[1], "000002.SZ", "10", "10", pre_close="10"),
            ),
            trading_calendar=calendar,
            universe_by_date={
                calendar[0]: ("920001.BJ", "000003.SZ", "000001.SZ", "000002.SZ")
            },
            benchmark_returns={calendar[1]: Decimal("0")},
        )

        self.assertEqual([order.stock_id for order in report.orders], ["000002.SZ"])

    def test_gap_up_changes_execution_fill_not_t_plus_one_order_or_cash_safety(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))
        config = BacktestConfig(
            run_id="gap-up-cash",
            top_n=1,
            initial_cash=Decimal("10000"),
            commission_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
        common: _RunInputs = {
            "signals": (
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            "trading_calendar": calendar,
            "universe_by_date": {calendar[0]: ("000001.SZ",)},
            "benchmark_returns": {calendar[1]: Decimal("0")},
        }
        flat_report = BacktestEngine(config).run(
            bars=(_bar(calendar[1], "000001.SZ", "10", "10"),),
            **common,
        )
        gap_up_report = BacktestEngine(config).run(
            bars=(_bar(calendar[1], "000001.SZ", "10.9", "10.9"),),
            **common,
        )

        self.assertEqual(flat_report.orders, gap_up_report.orders)
        self.assertEqual(flat_report.fills[0].quantity, 1000)
        self.assertEqual(gap_up_report.fills[0].quantity, 900)
        execution = cast(dict[str, object], gap_up_report.audit["execution"])
        self.assertEqual(execution["cash"], "190.0")
        self.assertGreaterEqual(Decimal(str(execution["cash"])), Decimal("0"))

    def test_real_backtest_fails_closed_without_verified_evaluation_manifest(self) -> None:
        engine = BacktestEngine(BacktestConfig(run_id="real", mode="real"))

        with self.assertRaisesRegex(FailClosedBacktestError, "verified evaluation manifest"):
            engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=date(2026, 1, 1),
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(_bar(date(2026, 1, 2), "000001.SZ", "10", "10"),),
                trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
                universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
                benchmark_returns={date(2026, 1, 2): Decimal("0")},
            )

    def test_contaminated_evaluation_manifest_does_not_claim_real_backtest_valid(self) -> None:
        report = BacktestEngine(
            BacktestConfig(
                run_id="contaminated",
                mode="real",
                top_n=1,
                initial_cash=Decimal("10000"),
                slippage_bps=Decimal("0"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=date(2026, 1, 1),
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(date(2026, 1, 2), "000001.SZ", "10", "10"),),
            trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
            universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
            benchmark_returns={date(2026, 1, 2): Decimal("0")},
            evaluation_manifest_sha256="eval-sha",
            evaluation_manifest_verified=True,
        )

        self.assertFalse(report.claims["real_backtest_valid"])
        self.assertTrue(report.claims["negative_result"])
        self.assertEqual(report.audit["evaluation_manifest_sha256"], "eval-sha")
        self.assertIsNone(report.audit["final_blind_manifest_sha256"])
        self.assertIn(
            "missing_verified_final_blind_manifest",
            report.audit["negative_result_reasons"],
        )

    def test_legacy_blind_manifest_parameters_cannot_bypass_final_blind_claim(self) -> None:
        report = BacktestEngine(
            BacktestConfig(
                run_id="legacy",
                mode="real",
                top_n=1,
                initial_cash=Decimal("10000"),
                slippage_bps=Decimal("0"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=date(2026, 1, 1),
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(date(2026, 1, 2), "000001.SZ", "10", "10"),),
            trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
            universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
            benchmark_returns={date(2026, 1, 2): Decimal("0")},
            blind_manifest_sha256="legacy-sha",
            blind_manifest_verified=True,
        )

        self.assertFalse(report.claims["real_backtest_valid"])
        self.assertTrue(report.claims["negative_result"])
        self.assertEqual(report.audit["evaluation_manifest_sha256"], "legacy-sha")
        self.assertIn(
            "legacy_blind_manifest_treated_as_evaluation_manifest_only",
            report.audit["negative_result_reasons"],
        )

    def test_real_backtest_fails_closed_on_missing_execution_bar(self) -> None:
        engine = BacktestEngine(BacktestConfig(run_id="missing-bar", mode="real", top_n=1))

        with self.assertRaisesRegex(FailClosedBacktestError, "missing execution bar"):
            engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=date(2026, 1, 1),
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(_bar(date(2026, 1, 3), "000001.SZ", "10", "10"),),
                trading_calendar=(date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)),
                universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
                benchmark_returns={date(2026, 1, 2): Decimal("0")},
                evaluation_manifest_sha256="eval-sha",
                evaluation_manifest_verified=True,
            )

    def test_real_backtest_fails_closed_when_fill_rate_below_threshold(self) -> None:
        engine = BacktestEngine(
            BacktestConfig(
                run_id="low-fill",
                mode="real",
                top_n=1,
                initial_cash=Decimal("10000"),
                min_real_fill_rate=Decimal("1"),
            )
        )

        with self.assertRaisesRegex(FailClosedBacktestError, "fill_rate") as raised:
            engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=date(2026, 1, 1),
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(_bar(date(2026, 1, 2), "000001.SZ", "11", "11"),),
                trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
                universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
                benchmark_returns={date(2026, 1, 2): Decimal("0")},
                evaluation_manifest_sha256="eval-sha",
                evaluation_manifest_verified=True,
            )
        self.assertIsNotNone(raised.exception.report)
        report = raised.exception.report
        assert report is not None
        self.assertFalse(report.claims["real_backtest_valid"])
        self.assertTrue(report.claims["negative_result"])
        self.assertEqual(report.audit["execution"]["order_count"], 1)
        self.assertEqual(report.audit["execution"]["blocked_reasons"]["limit_up_buy_blocked"], 1)
        self.assertFalse(report.audit["real_execution_quality"]["passed"])
        self.assertIn("fill_rate", str(report.audit["real_execution_quality"]["failure_reason"]))

    def test_real_backtest_fails_closed_when_untradable_rate_exceeds_threshold(self) -> None:
        engine = BacktestEngine(
            BacktestConfig(
                run_id="high-untradable",
                mode="real",
                top_n=1,
                initial_cash=Decimal("10000"),
                min_real_fill_rate=Decimal("0"),
                max_real_untradable_order_rate=Decimal("0"),
            )
        )

        with self.assertRaisesRegex(FailClosedBacktestError, "untradable_order_rate"):
            engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=date(2026, 1, 1),
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(_bar(date(2026, 1, 2), "000001.SZ", "11", "11"),),
                trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
                universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
                benchmark_returns={date(2026, 1, 2): Decimal("0")},
                evaluation_manifest_sha256="eval-sha",
                evaluation_manifest_verified=True,
            )

    def test_blocks_limit_up_buy_capacity_and_late_prediction(self) -> None:
        engine = BacktestEngine(BacktestConfig(run_id="blocked", top_n=1))
        with self.assertRaisesRegex(FailClosedBacktestError, "after as_of_date"):
            engine.run(
                signals=(
                    PredictionSignal(
                        as_of_date=date(2026, 1, 1),
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 2, 9, 0),
                    ),
                ),
                bars=(_bar(date(2026, 1, 2), "000001.SZ", "11", "11"),),
                trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
                universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
                benchmark_returns={date(2026, 1, 2): Decimal("0")},
            )

        report = engine.run(
            signals=(
                PredictionSignal(
                    as_of_date=date(2026, 1, 1),
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(date(2026, 1, 2), "000001.SZ", "11", "11", volume="100"),),
            trading_calendar=(date(2026, 1, 1), date(2026, 1, 2)),
            universe_by_date={date(2026, 1, 1): ("000001.SZ",)},
            benchmark_returns={date(2026, 1, 2): Decimal("0")},
        )

        self.assertTrue(report.fills[0].blocked)
        self.assertIn(report.fills[0].blocked_reason, {"capacity_exceeded", "limit_up_buy_blocked"})

    def test_marks_missing_holding_bar_with_previous_close_fallback(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 5))
        report = BacktestEngine(
            BacktestConfig(
                run_id="mtm-fallback",
                top_n=1,
                initial_cash=Decimal("10000"),
                slippage_bps=Decimal("0"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "000001.SZ", "10", "10"),),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ",)},
            benchmark_returns={calendar[1]: Decimal("0"), calendar[2]: Decimal("0")},
            confirmed_suspensions={calendar[2]: ("000001.SZ",)},
        )

        self.assertEqual(report.metrics["total_return"], "0")
        self.assertEqual(
            report.audit["mark_to_market_fallbacks"],
            [
                {
                    "trade_date": calendar[2].isoformat(),
                    "stock_id": "000001.SZ",
                    "fallback_trade_date": calendar[1].isoformat(),
                    "fallback_close": "10",
                    "quantity": 1000,
                    "reason": "confirmed_suspension_no_daily_bar_valued_at_previous_close",
                }
            ],
        )

    def test_missing_holding_bar_without_suspension_evidence_fails_closed(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 5))

        with self.assertRaisesRegex(
            FailClosedBacktestError, "without confirmed suspension evidence"
        ):
            BacktestEngine(
                BacktestConfig(
                    run_id="mtm-missing-evidence",
                    top_n=1,
                    initial_cash=Decimal("10000"),
                    slippage_bps=Decimal("0"),
                    commission_bps=Decimal("0"),
                    stamp_tax_bps=Decimal("0"),
                )
            ).run(
                signals=(
                    PredictionSignal(
                        as_of_date=calendar[0],
                        stock_id="000001.SZ",
                        score=Decimal("1"),
                        available_time=datetime(2026, 1, 1, 18, 0),
                    ),
                ),
                bars=(_bar(calendar[1], "000001.SZ", "10", "10"),),
                trading_calendar=calendar,
                universe_by_date={calendar[0]: ("000001.SZ",)},
                benchmark_returns={calendar[1]: Decimal("0"), calendar[2]: Decimal("0")},
            )

    def test_confirmed_suspension_blocks_missing_execution_without_dropping_order(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))
        report = BacktestEngine(
            BacktestConfig(
                run_id="confirmed-suspension",
                mode="real",
                top_n=1,
                min_real_fill_rate=Decimal("0"),
                max_real_untradable_order_rate=Decimal("1"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "000002.SZ", "10", "10"),),
            trading_calendar=calendar,
            universe_by_date={calendar[0]: ("000001.SZ",)},
            benchmark_returns={calendar[1]: Decimal("0")},
            evaluation_manifest_sha256="eval-sha",
            evaluation_manifest_verified=True,
            confirmed_suspensions={calendar[1]: ("000001.SZ",)},
        )

        self.assertEqual(len(report.orders), 1)
        self.assertEqual(report.fills[0].blocked_reason, "confirmed_suspension_no_daily_bar")
        execution = cast(dict[str, object], report.audit["execution"])
        self.assertEqual(execution["filled_order_count"], 0)
        self.assertEqual(execution["blocked_order_count"], 1)

    def test_board_specific_limit_rules_keep_config_fallback(self) -> None:
        calendar = (date(2026, 1, 1), date(2026, 1, 2))
        common = {
            "trading_calendar": calendar,
            "benchmark_returns": {calendar[1]: Decimal("0")},
        }

        main_report = BacktestEngine(
            BacktestConfig(run_id="main-limit", top_n=1, initial_cash=Decimal("10000"))
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "000001.SZ", "11", "11"),),
            universe_by_date={calendar[0]: ("000001.SZ",)},
            **common,
        )
        growth_report = BacktestEngine(
            BacktestConfig(run_id="growth-limit", top_n=1, initial_cash=Decimal("10000"))
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="300001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "300001.SZ", "11", "11"),),
            universe_by_date={calendar[0]: ("300001.SZ",)},
            **common,
        )
        fallback_report = BacktestEngine(
            BacktestConfig(
                run_id="fallback-limit",
                top_n=1,
                initial_cash=Decimal("10000"),
                limit_up_pct=Decimal("0.05"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="200001.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
            ),
            bars=(_bar(calendar[1], "200001.SZ", "10.6", "10.6"),),
            universe_by_date={calendar[0]: ("200001.SZ",)},
            **common,
        )

        self.assertEqual(main_report.fills[0].blocked_reason, "limit_up_buy_blocked")
        self.assertFalse(growth_report.fills[0].blocked)
        self.assertEqual(fallback_report.fills[0].blocked_reason, "limit_up_buy_blocked")

    def test_rebalance_targets_weights_instead_of_rebuying_existing_positions(self) -> None:
        calendar = (
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 5),
            date(2026, 1, 6),
        )
        report = BacktestEngine(
            BacktestConfig(
                run_id="target-weight",
                top_n=2,
                initial_cash=Decimal("10000"),
                slippage_bps=Decimal("0"),
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
            )
        ).run(
            signals=(
                PredictionSignal(
                    as_of_date=calendar[0],
                    stock_id="000001.SZ",
                    score=Decimal("2"),
                    available_time=datetime(2026, 1, 1, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[1],
                    stock_id="000001.SZ",
                    score=Decimal("2"),
                    available_time=datetime(2026, 1, 2, 18, 0),
                ),
                PredictionSignal(
                    as_of_date=calendar[1],
                    stock_id="000002.SZ",
                    score=Decimal("1"),
                    available_time=datetime(2026, 1, 2, 18, 0),
                ),
            ),
            bars=(
                _bar(calendar[1], "000001.SZ", "10", "10"),
                _bar(calendar[2], "000001.SZ", "10", "10"),
                _bar(calendar[2], "000002.SZ", "10", "10"),
                _bar(calendar[3], "000001.SZ", "10", "10"),
                _bar(calendar[3], "000002.SZ", "10", "10"),
            ),
            trading_calendar=calendar,
            universe_by_date={
                calendar[0]: ("000001.SZ",),
                calendar[1]: ("000001.SZ", "000002.SZ"),
            },
            benchmark_returns={
                calendar[1]: Decimal("0"),
                calendar[2]: Decimal("0"),
                calendar[3]: Decimal("0"),
            },
        )

        rebalance_orders = [order for order in report.orders if order.as_of_date == calendar[1]]
        self.assertEqual(
            [(order.stock_id, order.side, order.requested_quantity) for order in rebalance_orders],
            [("000001.SZ", "sell", 500), ("000002.SZ", "buy", 0)],
        )
        self.assertEqual(rebalance_orders[1].target_notional, Decimal("5000"))
        self.assertEqual(report.audit["execution"]["fill_rate"], "1")
        self.assertEqual(report.audit["execution"]["gross_exposure"], "10000")
        self.assertEqual(report.audit["execution"]["position_concentration"], "0.5")


if __name__ == "__main__":
    unittest.main()
