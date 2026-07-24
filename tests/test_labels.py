import unittest
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

from seatflow.data.contracts import AdjustmentFactorEvidence, DailyBar
from seatflow.labels import (
    FIXTURE_ADJUSTMENT_POLICY,
    FIXTURE_CALENDAR_POLICY,
    FROZEN_ADJUSTMENT_POLICY,
    FROZEN_CALENDAR_POLICY,
    ExecutablePriceEvidence,
    LabelingConfig,
    MarketDispositionEvidence,
    executable_open_evidence_from_daily_bar,
    generate_executable_excess_return_labels_v2,
    generate_future_return_labels,
    load_labeling_config,
)


def _bar(day: int, close: str, available_hour: int = 18) -> DailyBar:
    trade_date = date(2026, 1, day)
    return DailyBar(
        trade_date=trade_date,
        stock_id="000001.SZ",
        open_price=Decimal(close),
        high_price=Decimal(close),
        low_price=Decimal(close),
        close_price=Decimal(close),
        volume=Decimal("100"),
        amount=Decimal("1000"),
        source_name="fixture",
        available_time=datetime.combine(trade_date, time(available_hour, 0)),
    )


class LabelGenerationTest(unittest.TestCase):
    def test_future_return_labels_start_after_cutoff_date(self) -> None:
        bars = tuple(_bar(day, str(10 + day)) for day in range(1, 7))
        cutoff = datetime(2026, 1, 2, 18, 0)

        labels = generate_future_return_labels(
            bars,
            as_of_date=date(2026, 1, 2),
            cutoff_time=cutoff,
            target_stock_ids=("000001.SZ",),
            horizons=(1, 3),
            trading_calendar=tuple(bar.trade_date for bar in bars),
        )

        self.assertEqual([label.horizon for label in labels], [1, 3])
        self.assertTrue(all(label.label_start_date > cutoff.date() for label in labels))
        self.assertEqual(labels[0].future_return, Decimal("1") / Decimal("12"))
        self.assertEqual(labels[0].direction_label, 1)

    def test_rejects_feature_bar_unavailable_at_cutoff(self) -> None:
        bars = (_bar(1, "10"), _bar(2, "11", available_hour=19), _bar(3, "12"))

        with self.assertRaisesRegex(ValueError, "available_time is after cutoff_time"):
            generate_future_return_labels(
                bars,
                as_of_date=date(2026, 1, 2),
                cutoff_time=datetime(2026, 1, 2, 18, 0),
                target_stock_ids=("000001.SZ",),
                horizons=(1,),
                trading_calendar=tuple(bar.trade_date for bar in bars),
            )

    def test_rejects_future_label_that_would_be_available_before_cutoff(self) -> None:
        bars = (
            _bar(1, "10"),
            _bar(2, "11"),
            DailyBar(
                trade_date=date(2026, 1, 3),
                stock_id="000001.SZ",
                open_price=Decimal("12"),
                high_price=Decimal("12"),
                low_price=Decimal("12"),
                close_price=Decimal("12"),
                volume=Decimal("100"),
                amount=Decimal("1000"),
                source_name="fixture",
                available_time=datetime(2026, 1, 2, 17, 0),
            ),
        )

        with self.assertRaisesRegex(ValueError, "future label availability"):
            generate_future_return_labels(
                bars,
                as_of_date=date(2026, 1, 2),
                cutoff_time=datetime(2026, 1, 2, 18, 0),
                target_stock_ids=("000001.SZ",),
                horizons=(1,),
                trading_calendar=tuple(bar.trade_date for bar in bars),
            )

    def test_research_labeling_requires_frozen_adjustment_rules(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen adjustment policy"):
            LabelingConfig(
                horizons=(1, 3, 5),
                adjustment_policy=FIXTURE_ADJUSTMENT_POLICY,
                calendar_policy=FROZEN_CALENDAR_POLICY,
            ).require_research_ready()

    def test_frozen_calendar_policy_requires_explicit_trading_calendar(self) -> None:
        bars = tuple(_bar(day, str(10 + day)) for day in range(1, 4))

        with self.assertRaisesRegex(ValueError, "frozen trading calendar"):
            generate_future_return_labels(
                bars,
                as_of_date=date(2026, 1, 2),
                cutoff_time=datetime(2026, 1, 2, 18, 0),
                target_stock_ids=("000001.SZ",),
                horizons=(1,),
            )

    def test_fixture_policy_is_not_research_ready(self) -> None:
        config = LabelingConfig(
            horizons=(1, 3, 5),
            adjustment_policy=FIXTURE_ADJUSTMENT_POLICY,
            calendar_policy=FIXTURE_CALENDAR_POLICY,
        )

        self.assertFalse(config.research_ready)

    def test_default_labeling_config_is_research_ready_for_one_three_five(self) -> None:
        config = load_labeling_config(Path("configs/labeling.toml"))

        self.assertEqual(config.horizons, (1, 3, 5))
        self.assertEqual(config.adjustment_policy, FROZEN_ADJUSTMENT_POLICY)
        self.assertEqual(config.calendar_policy, FROZEN_CALENDAR_POLICY)
        self.assertTrue(config.research_ready)

    def test_v2_uses_t_plus_one_open_and_benchmark_excess_return(self) -> None:
        calendar = tuple(date(2026, 1, day) for day in range(1, 8))
        evidence = _v2_prices(calendar)

        labels = generate_executable_excess_return_labels_v2(
            as_of_date=calendar[0],
            cutoff_time=datetime(2026, 1, 1, 18, 0),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            trading_calendar=calendar,
            benchmark_by_stock={"000001.SZ": "000300.SH"},
            stock_open_prices=evidence[0],
            benchmark_open_prices=evidence[1],
            neutral_band=Decimal("0.01"),
        )

        self.assertEqual(len(labels), 1)
        label = labels[0]
        self.assertEqual(label.label_start_date, calendar[1])
        self.assertEqual(label.label_end_date, calendar[2])
        self.assertEqual(label.stock_return, Decimal("0.1"))
        self.assertEqual(label.benchmark_return, Decimal("0.02"))
        self.assertEqual(label.excess_return, Decimal("0.08"))
        self.assertEqual(label.direction_label, 1)
        self.assertEqual(label.disposition, "eligible")
        self.assertGreater(label.label_available_time, label.cutoff_time)

    def test_v2_records_non_executable_disposition_without_numeric_label(self) -> None:
        calendar = tuple(date(2026, 1, day) for day in range(1, 8))
        evidence = _v2_prices(calendar)

        label = generate_executable_excess_return_labels_v2(
            as_of_date=calendar[0],
            cutoff_time=datetime(2026, 1, 1, 18, 0),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            trading_calendar=calendar,
            benchmark_by_stock={"000001.SZ": "000300.SH"},
            stock_open_prices=evidence[0],
            benchmark_open_prices=evidence[1],
            dispositions={
                ("000001.SZ", calendar[1]): MarketDispositionEvidence(
                    disposition="suspended", source_hash="b" * 64
                )
            },
        )[0]

        self.assertEqual(label.disposition, "suspended")
        self.assertIsNone(label.excess_return)
        self.assertIsNone(label.direction_label)

    def test_v2_records_unobserved_horizon_without_numeric_label(self) -> None:
        calendar = tuple(date(2026, 1, day) for day in range(1, 8))
        label = generate_executable_excess_return_labels_v2(
            as_of_date=calendar[0],
            cutoff_time=datetime(2026, 1, 1, 18, 0),
            target_stock_ids=("000001.SZ",),
            horizons=(5,),
            trading_calendar=calendar,
            benchmark_by_stock={"000001.SZ": "000300.SH"},
            stock_open_prices={},
            benchmark_open_prices={},
            dispositions={
                ("000001.SZ", calendar[6]): MarketDispositionEvidence(
                    disposition="unobserved_horizon", source_hash="d" * 64
                )
            },
        )[0]

        self.assertEqual(label.disposition, "unobserved_horizon")
        self.assertIsNone(label.excess_return)

    def test_v2_includes_market_status_hash_for_eligible_labels(self) -> None:
        calendar = tuple(date(2026, 1, day) for day in range(1, 8))
        evidence = _v2_prices(calendar)
        base = generate_executable_excess_return_labels_v2(
            as_of_date=calendar[0],
            cutoff_time=datetime(2026, 1, 1, 18, 0),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            trading_calendar=calendar,
            benchmark_by_stock={"000001.SZ": "000300.SH"},
            stock_open_prices=evidence[0],
            benchmark_open_prices=evidence[1],
        )[0]
        with_status = generate_executable_excess_return_labels_v2(
            as_of_date=calendar[0],
            cutoff_time=datetime(2026, 1, 1, 18, 0),
            target_stock_ids=("000001.SZ",),
            horizons=(1,),
            trading_calendar=calendar,
            benchmark_by_stock={"000001.SZ": "000300.SH"},
            stock_open_prices=evidence[0],
            benchmark_open_prices=evidence[1],
            market_status_hashes={"000001.SZ": "c" * 64},
        )[0]

        self.assertNotEqual(base.source_hash, with_status.source_hash)

    def test_v2_open_evidence_binds_actual_adjustment_factor_to_daily_bar(self) -> None:
        bar = _bar(2, "11")
        adjustment = AdjustmentFactorEvidence(
            stock_id="000001.SZ",
            trade_date=bar.trade_date,
            adjustment_factor=Decimal("1.25"),
            source="tushare:adj_factor",
            retrieved_at=datetime(2026, 1, 3, tzinfo=UTC),
            row_payload_sha256="f" * 64,
        )

        evidence = executable_open_evidence_from_daily_bar(bar, adjustment)

        self.assertEqual(evidence.price, bar.open_price)
        self.assertEqual(evidence.adjustment_factor, Decimal("1.25"))
        self.assertRegex(evidence.source_hash, r"^[0-9a-f]{64}$")


def _v2_prices(
    calendar: tuple[date, ...],
) -> tuple[
    dict[tuple[str, date], ExecutablePriceEvidence],
    dict[tuple[str, date], ExecutablePriceEvidence],
]:
    stock = {
        ("000001.SZ", calendar[1]): _v2_evidence(calendar[1], "10", "1", "a"),
        ("000001.SZ", calendar[2]): _v2_evidence(calendar[2], "11", "1", "c"),
    }
    benchmark = {
        ("000300.SH", calendar[1]): _v2_evidence(calendar[1], "100", "1", "d"),
        ("000300.SH", calendar[2]): _v2_evidence(calendar[2], "102", "1", "e"),
    }
    return stock, benchmark


def _v2_evidence(day: date, price: str, adjustment: str, seed: str) -> ExecutablePriceEvidence:
    return ExecutablePriceEvidence(
        trade_date=day,
        price=Decimal(price),
        adjustment_factor=Decimal(adjustment),
        available_time=datetime.combine(day, time(18, 0)),
        source_hash=seed * 64,
    )


if __name__ == "__main__":
    unittest.main()
