"""Tests for tradeability classification (Task 12).

A sample is only *backtest-eligible* if the stock was actually tradable at
the entry point and throughout the 5-day horizon. ``classify_tradeability``
inspects the OHLC bar for the entry date and the subsequent bars and
returns one of the :class:`TradeabilityState` values.

States
------
- ``TRADABLE``: the stock traded normally on the entry date and remained
  listed for the full horizon.
- ``SUSPENDED_AT_ENTRY``: the stock was suspended on the entry date (no
  trade executed — volume == 0 and a ``suspended`` flag, or open == close
  with ``volume == 0``).
- ``ONE_PRICE_LIMIT_UP`` / ``ONE_PRICE_LIMIT_DOWN``: the entry-day bar is a
  one-price limit bar (open == close == high == low at the limit price),
  meaning the stock could not realistically be bought at the recorded
  price.
- ``MISSING_ENTRY_PRICE``: no OHLC bar exists for the entry date.
- ``DELISTED_DURING_HORIZON``: the stock was delisted within the 5-day
  horizon (fewer than 5 bars available after the entry date, or an explicit
  ``delisted`` marker).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


from seatflow.labels.tradeability import (
    TradeabilityState,
    classify_tradeability,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _bar(
    date: str,
    *,
    open_: float,
    close: float,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1.0e6,
    suspended: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    bar: dict[str, Any] = {
        "date": _utc(date),
        "open": float(open_),
        "close": float(close),
        "high": float(high if high is not None else max(open_, close)),
        "low": float(low if low is not None else min(open_, close)),
        "volume": float(volume),
        "suspended": bool(suspended),
    }
    bar.update(extra)
    return bar


def _normal_5day(start: str = "2026-01-05T00:00:00+00:00") -> list[dict[str, Any]]:
    """Five normal trading days starting at *start*."""
    return [
        _bar("2026-01-05T00:00:00+00:00", open_=10, close=10.5),
        _bar("2026-01-06T00:00:00+00:00", open_=10.5, close=11),
        _bar("2026-01-07T00:00:00+00:00", open_=11, close=10.8),
        _bar("2026-01-08T00:00:00+00:00", open_=10.8, close=11.2),
        _bar("2026-01-09T00:00:00+00:00", open_=11.2, close=11.5),
    ]


# ── Enum contract ──────────────────────────────────────────────────────


class TestTradeabilityEnum:
    def test_enum_members_present(self):
        assert TradeabilityState.TRADABLE.value == "TRADABLE"
        assert TradeabilityState.SUSPENDED_AT_ENTRY.value == "SUSPENDED_AT_ENTRY"
        assert TradeabilityState.ONE_PRICE_LIMIT_UP.value == "ONE_PRICE_LIMIT_UP"
        assert TradeabilityState.ONE_PRICE_LIMIT_DOWN.value == "ONE_PRICE_LIMIT_DOWN"
        assert TradeabilityState.MISSING_ENTRY_PRICE.value == "MISSING_ENTRY_PRICE"
        assert TradeabilityState.DELISTED_DURING_HORIZON.value == "DELISTED_DURING_HORIZON"


# ── Classification ─────────────────────────────────────────────────────


class TestClassifyTradeability:
    def test_valid_tradable(self):
        """Normal trading on entry date and full 5-day horizon → TRADABLE."""
        prices = _normal_5day()
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.TRADABLE

    def test_suspended_at_entry(self):
        """Stock suspended (volume == 0) on the entry date."""
        prices = _normal_5day()
        prices[0] = _bar(
            "2026-01-05T00:00:00+00:00",
            open_=10,
            close=10,
            high=10,
            low=10,
            volume=0.0,
            suspended=True,
        )
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.SUSPENDED_AT_ENTRY

    def test_one_price_limit_up(self):
        """Entry day opens at the upper limit and stays there (one-price)."""
        prices = _normal_5day()
        # A one-price limit-up bar: open == close == high == low, positive
        # limit, and a limit_up flag so it is distinguishable from a
        # suspended bar (which also collapses to one price but with volume 0).
        prices[0] = _bar(
            "2026-01-05T00:00:00+00:00",
            open_=11.0,
            close=11.0,
            high=11.0,
            low=11.0,
            volume=5.0e6,
            limit_up=True,
        )
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.ONE_PRICE_LIMIT_UP

    def test_one_price_limit_down(self):
        """Entry day opens at the lower limit and stays there (one-price)."""
        prices = _normal_5day()
        prices[0] = _bar(
            "2026-01-05T00:00:00+00:00",
            open_=9.0,
            close=9.0,
            high=9.0,
            low=9.0,
            volume=5.0e6,
            limit_down=True,
        )
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.ONE_PRICE_LIMIT_DOWN

    def test_delisted_during_horizon(self):
        """Fewer than 5 bars after entry → delisted during horizon."""
        prices = [
            _bar("2026-01-05T00:00:00+00:00", open_=10, close=10.5),
            _bar("2026-01-06T00:00:00+00:00", open_=10.5, close=11),
            _bar("2026-01-07T00:00:00+00:00", open_=11, close=10.8, delisted=True),
        ]
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.DELISTED_DURING_HORIZON

    def test_missing_entry_price(self):
        """No OHLC bar for the entry date → MISSING_ENTRY_PRICE."""
        prices = _normal_5day()
        # Entry date is not the first bar's date.
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2025-01-01T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.MISSING_ENTRY_PRICE

    def test_empty_prices_df_is_missing_entry(self):
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=[],
        )
        assert state == TradeabilityState.MISSING_ENTRY_PRICE

    def test_suspended_takes_precedence_over_delisted(self):
        """A suspended entry day with fewer than 5 bars is still SUSPENDED."""
        prices = [
            _bar(
                "2026-01-05T00:00:00+00:00",
                open_=10,
                close=10,
                high=10,
                low=10,
                volume=0.0,
                suspended=True,
            ),
        ]
        state = classify_tradeability(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert state == TradeabilityState.SUSPENDED_AT_ENTRY
