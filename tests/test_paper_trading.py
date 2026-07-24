from __future__ import annotations

import tempfile
import time
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from seatflow.paper_trading import (
    JsonPaperBroker,
    MarketQuote,
    PaperBrokerError,
    PaperOrder,
    PaperSignal,
    PaperTradingConfig,
    PaperTradingService,
    QuoteBatch,
    SignalBatch,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        account_id="test-paper",
        initial_cash=Decimal("100000"),
        top_n=2,
        minimum_score=Decimal("0.5"),
        cash_buffer_pct=Decimal("0"),
        max_position_pct=Decimal("0.5"),
        commission_bps=Decimal("0"),
        minimum_commission=Decimal("0"),
        stamp_tax_bps=Decimal("0"),
        slippage_bps=Decimal("0"),
        minimum_lot_size=100,
    )


def _signals(day: date, *items: tuple[str, str]) -> SignalBatch:
    return SignalBatch(
        as_of_date=day,
        generated_at=datetime(day.year, day.month, day.day, 18, 0, tzinfo=SHANGHAI),
        signals=tuple(PaperSignal(stock_id, Decimal(score)) for stock_id, score in items),
    )


def _quotes(day: date, *items: tuple[str, str]) -> QuoteBatch:
    return QuoteBatch(
        trade_date=day,
        captured_at=datetime(day.year, day.month, day.day, 9, 31, tzinfo=SHANGHAI),
        quotes=tuple(MarketQuote(stock_id, Decimal(price)) for stock_id, price in items),
    )


class PaperTradingTest(unittest.TestCase):
    def test_service_buys_replays_and_rebalances_on_next_day(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "account.json"
            config = _config()
            broker = JsonPaperBroker(state_path, config)
            broker.initialize()
            service = PaperTradingService(broker, config)

            first = service.run(
                cycle_id="day-1",
                signal_batch=_signals(
                    date(2026, 7, 17),
                    ("600000.SH", "0.9"),
                    ("000001.SZ", "0.8"),
                    ("300750.SZ", "0.4"),
                ),
                quote_batch=_quotes(
                    date(2026, 7, 20),
                    ("600000.SH", "10"),
                    ("000001.SZ", "20"),
                    ("300750.SZ", "50"),
                ),
            )

            first_positions = first.account_after.position_map()
            self.assertEqual(first_positions["600000.SH"].quantity, 5000)
            self.assertEqual(first_positions["000001.SZ"].quantity, 2500)
            self.assertEqual(first.account_after.cash, Decimal("0"))

            replay = service.run(
                cycle_id="day-1",
                signal_batch=_signals(
                    date(2026, 7, 17),
                    ("600000.SH", "0.9"),
                    ("000001.SZ", "0.8"),
                    ("300750.SZ", "0.4"),
                ),
                quote_batch=_quotes(
                    date(2026, 7, 20),
                    ("600000.SH", "10"),
                    ("000001.SZ", "20"),
                    ("300750.SZ", "50"),
                ),
            )
            self.assertTrue(replay.sell_batch.replayed)
            self.assertTrue(replay.buy_batch.replayed)
            self.assertEqual(replay.account_after.to_dict(), first.account_after.to_dict())

            with self.assertRaisesRegex(PaperBrokerError, "different request"):
                service.run(
                    cycle_id="day-1",
                    signal_batch=_signals(
                        date(2026, 7, 17),
                        ("600000.SH", "0.91"),
                        ("000001.SZ", "0.8"),
                    ),
                    quote_batch=_quotes(
                        date(2026, 7, 20),
                        ("600000.SH", "10"),
                        ("000001.SZ", "20"),
                        ("300750.SZ", "50"),
                    ),
                )

            second = service.run(
                cycle_id="day-2",
                signal_batch=_signals(
                    date(2026, 7, 20),
                    ("000001.SZ", "0.95"),
                    ("300750.SZ", "0.85"),
                ),
                quote_batch=_quotes(
                    date(2026, 7, 21),
                    ("600000.SH", "11"),
                    ("000001.SZ", "20"),
                    ("300750.SZ", "50"),
                ),
            )

            second_positions = second.account_after.position_map()
            self.assertNotIn("600000.SH", second_positions)
            self.assertIn("000001.SZ", second_positions)
            self.assertIn("300750.SZ", second_positions)
            self.assertEqual(second_positions["300750.SZ"].quantity % 100, 0)
            self.assertEqual(second.sell_batch.fills[0].status, "filled")

    def test_broker_enforces_t_plus_one_and_limit_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "account.json"
            config = _config()
            broker = JsonPaperBroker(state_path, config)
            broker.initialize()
            day = date(2026, 7, 20)
            quote = MarketQuote("600000.SH", Decimal("10"))
            quote_map = {quote.stock_id: quote}

            buy = broker.submit_market_orders(
                batch_id="buy",
                trade_date=day,
                orders=(PaperOrder("600000.SH", "buy", 100, "test"),),
                quotes=quote_map,
            )
            self.assertEqual(buy.fills[0].status, "filled")

            same_day_sell = broker.submit_market_orders(
                batch_id="same-day-sell",
                trade_date=day,
                orders=(PaperOrder("600000.SH", "sell", 100, "test"),),
                quotes=quote_map,
            )
            self.assertEqual(same_day_sell.fills[0].status, "rejected")
            self.assertEqual(same_day_sell.fills[0].message, "t_plus_one_sell_blocked")

            next_day_sell = broker.submit_market_orders(
                batch_id="next-day-sell",
                trade_date=date(2026, 7, 21),
                orders=(PaperOrder("600000.SH", "sell", 100, "test"),),
                quotes=quote_map,
            )
            self.assertEqual(next_day_sell.fills[0].status, "filled")
            self.assertFalse(next_day_sell.account.positions)

            limit_quote = MarketQuote("000001.SZ", Decimal("20"), limit_up=True)
            blocked_buy = broker.submit_market_orders(
                batch_id="limit-buy",
                trade_date=date(2026, 7, 21),
                orders=(PaperOrder("000001.SZ", "buy", 100, "test"),),
                quotes={limit_quote.stock_id: limit_quote},
            )
            self.assertEqual(blocked_buy.fills[0].message, "limit_up_buy_blocked")

            with self.assertRaisesRegex(PaperBrokerError, "before the last trade date"):
                broker.submit_market_orders(
                    batch_id="out-of-order",
                    trade_date=day,
                    orders=(),
                    quotes={},
                )

    def test_stale_lock_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / "account.json"
            config = _config()
            broker = JsonPaperBroker(state_path, config)
            broker.initialize()
            lock_path = state_path.with_suffix(".json.lock")
            lock_path.write_text("stale", encoding="ascii")
            stale_time = time.time() - 301
            lock_path.touch()
            import os

            os.utime(lock_path, (stale_time, stale_time))

            result = broker.submit_market_orders(
                batch_id="after-stale-lock",
                trade_date=date(2026, 7, 20),
                orders=(),
                quotes={},
            )

            self.assertFalse(result.replayed)
            self.assertFalse(lock_path.exists())

    def test_non_zero_costs_and_slippage_change_cash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = PaperTradingConfig(
                account_id="cost-test",
                initial_cash=Decimal("10000"),
                top_n=1,
                minimum_score=Decimal("0"),
                cash_buffer_pct=Decimal("0"),
                max_position_pct=Decimal("1"),
                commission_bps=Decimal("3"),
                minimum_commission=Decimal("5"),
                stamp_tax_bps=Decimal("5"),
                slippage_bps=Decimal("5"),
                minimum_lot_size=100,
            )
            broker = JsonPaperBroker(Path(temporary_dir) / "account.json", config)
            broker.initialize()
            quote = MarketQuote("600000.SH", Decimal("10"))

            buy = broker.submit_market_orders(
                batch_id="cost-buy",
                trade_date=date(2026, 7, 20),
                orders=(PaperOrder("600000.SH", "buy", 100, "test"),),
                quotes={quote.stock_id: quote},
            )

            self.assertEqual(buy.fills[0].price, Decimal("10.005"))
            self.assertEqual(buy.fills[0].commission, Decimal("5"))
            self.assertEqual(buy.account.cash, Decimal("8994.500"))

    def test_service_rejects_same_day_signal_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config()
            broker = JsonPaperBroker(Path(temporary_dir) / "account.json", config)
            broker.initialize()

            with self.assertRaisesRegex(ValueError, "earlier trading date"):
                PaperTradingService(broker, config).run(
                    cycle_id="invalid",
                    signal_batch=_signals(date(2026, 7, 20), ("600000.SH", "0.9")),
                    quote_batch=_quotes(date(2026, 7, 20), ("600000.SH", "10")),
                )

    def test_service_compares_signal_and_quote_as_absolute_times(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = _config()
            broker = JsonPaperBroker(Path(temporary_dir) / "account.json", config)
            broker.initialize()
            signal_batch = SignalBatch(
                as_of_date=date(2026, 7, 19),
                generated_at=datetime.fromisoformat("2026-07-19T23:30:00-12:00"),
                signals=(PaperSignal("600000.SH", Decimal("0.9")),),
            )

            with self.assertRaisesRegex(ValueError, "before the quote snapshot"):
                PaperTradingService(broker, config).run(
                    cycle_id="timezone-future",
                    signal_batch=signal_batch,
                    quote_batch=_quotes(date(2026, 7, 20), ("600000.SH", "10")),
                )


if __name__ == "__main__":
    unittest.main()
