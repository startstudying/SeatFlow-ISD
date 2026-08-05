"""Tests for multi-horizon return labels (Task 12).

A *return path* records the daily and cumulative returns for a stock from
an entry point over a 5-trading-day horizon. The labels are the prediction
targets:

- ``d1_return`` … ``d5_return``: simple daily returns Open-to-Close each day.
- ``cum_excess_1d`` / ``cum_excess_3d`` / ``cum_excess_5d``: exact compound
  excess returns over the market index.

Exactness contract
-------------------
The compound returns must be the *exact* product of the daily growth
factors, never a summed approximation. For prices ``[10, 11, 12, 11, 13, 14]``

    d1 = (11 - 10) / 10                       = 0.1
    d2 = (12 - 11) / 11                       = 1/11
    d3 = (11 - 12) / 12                       = -1/12
    d4 = (13 - 11) / 11                       = 2/11
    d5 = (14 - 13) / 13                       = 1/13

    cum_1d = d1                               = 0.1
    cum_3d = (1+d1)(1+d2)(1+d3) - 1
           = (11/10)(12/11)(11/12) - 1
           = 11/10 - 1                         = 0.1
    cum_5d = (1+d1)(1+d2)(1+d3)(1+d4)(1+d5) - 1
           = 14/10 - 1                         = 0.4
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from seatflow.labels.returns import (
    MarketExcessPath,
    ReturnPath,
    compute_market_excess_path,
    compute_return_path,
    cumulate_horizons,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _bar(date: str, open_: float, close: float, **extra: Any) -> dict[str, Any]:
    """Build one OHLC bar dict for a single trading day (midnight UTC)."""
    bar: dict[str, Any] = {
        "date": _utc(date),
        "open": float(open_),
        "close": float(close),
    }
    bar.update(extra)
    return bar


def _price_df(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A prices_df is an ordered list of OHLC bars keyed by trading date."""
    return bars


# ── Daily path computation ─────────────────────────────────────────────


class TestDailyPath:
    def test_daily_returns_open_to_close(self):
        """d1..d5 are simple Open-to-Close returns for each of the 5 days."""
        prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", open_=10, close=11),  # day 1
                _bar("2026-01-06T00:00:00+00:00", open_=11, close=12),  # day 2
                _bar("2026-01-07T00:00:00+00:00", open_=12, close=11),  # day 3
                _bar("2026-01-08T00:00:00+00:00", open_=11, close=13),  # day 4
                _bar("2026-01-09T00:00:00+00:00", open_=13, close=14),  # day 5
            ]
        )
        path = compute_return_path(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert isinstance(path, ReturnPath)
        assert path.stock_id == "000001.SZ"
        assert path.entry_price == pytest.approx(10.0)
        # d1 = (11-10)/10 = 0.1 exactly
        assert path.d1_return == pytest.approx(0.1, abs=1e-12)
        # d2 = (12-11)/11
        assert path.d2_return == pytest.approx(1.0 / 11.0, abs=1e-12)
        # d3 = (11-12)/12
        assert path.d3_return == pytest.approx(-1.0 / 12.0, abs=1e-12)
        # d4 = (13-11)/11
        assert path.d4_return == pytest.approx(2.0 / 11.0, abs=1e-12)
        # d5 = (14-13)/13
        assert path.d5_return == pytest.approx(1.0 / 13.0, abs=1e-12)

    def test_exit_prices_recorded(self):
        prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 10, 11),
                _bar("2026-01-06T00:00:00+00:00", 11, 12),
                _bar("2026-01-07T00:00:00+00:00", 12, 11),
                _bar("2026-01-08T00:00:00+00:00", 11, 13),
                _bar("2026-01-09T00:00:00+00:00", 13, 14),
            ]
        )
        path = compute_return_path(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert path.exit_1d_price == pytest.approx(11.0)
        assert path.exit_3d_price == pytest.approx(11.0)  # close of day 3
        assert path.exit_5d_price == pytest.approx(14.0)  # close of day 5

    def test_entry_price_is_day1_open(self):
        prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 42.0, 43.5),
                _bar("2026-01-06T00:00:00+00:00", 43.5, 44.0),
                _bar("2026-01-07T00:00:00+00:00", 44.0, 41.0),
                _bar("2026-01-08T00:00:00+00:00", 41.0, 45.0),
                _bar("2026-01-09T00:00:00+00:00", 45.0, 46.0),
            ]
        )
        path = compute_return_path(
            stock_id="000002.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )
        assert path.entry_price == pytest.approx(42.0)


# ── Exact compound horizons ────────────────────────────────────────────


class TestCompoundHorizons:
    def test_r1_equals_d1_exactly(self):
        """R1 (1-day compound) equals d1_return exactly."""
        daily = [0.1, 1.0 / 11.0, -1.0 / 12.0, 2.0 / 11.0, 1.0 / 13.0]
        r1 = cumulate_horizons(daily, h=1)
        assert r1 == pytest.approx(0.1, abs=1e-12)

    def test_r3_compound_from_d1_d2_d3(self):
        """R3 = (1+d1)(1+d2)(1+d3) - 1 within 1e-12."""
        daily = [0.1, 1.0 / 11.0, -1.0 / 12.0, 2.0 / 11.0, 1.0 / 13.0]
        r3 = cumulate_horizons(daily, h=3)
        expected = (1.1) * (12.0 / 11.0) * (11.0 / 12.0) - 1.0
        assert r3 == pytest.approx(expected, abs=1e-12)
        # The explicit closed form: (11/10)(12/11)(11/12) - 1 = 11/10 - 1 = 0.1
        assert r3 == pytest.approx(0.1, abs=1e-12)

    def test_r5_compound_from_all_five(self):
        """R5 = (1+d1)(1+d2)(1+d3)(1+d4)(1+d5) - 1 = 14/10 - 1 = 0.4."""
        daily = [0.1, 1.0 / 11.0, -1.0 / 12.0, 2.0 / 11.0, 1.0 / 13.0]
        r5 = cumulate_horizons(daily, h=5)
        expected = (14.0 / 10.0) - 1.0
        assert r5 == pytest.approx(expected, abs=1e-12)
        assert r5 == pytest.approx(0.4, abs=1e-12)

    def test_r3_uses_only_first_three(self):
        """Changing d4, d5 must not affect R3."""
        daily_a = [0.1, 0.05, -0.02, 0.3, 0.4]
        daily_b = [0.1, 0.05, -0.02, -0.5, -0.9]
        assert cumulate_horizons(daily_a, h=3) == pytest.approx(
            cumulate_horizons(daily_b, h=3), abs=1e-12
        )

    def test_invalid_horizon_raises(self):
        daily = [0.1, 0.2, 0.3, 0.4, 0.5]
        with pytest.raises(ValueError):
            cumulate_horizons(daily, h=2)
        with pytest.raises(ValueError):
            cumulate_horizons(daily, h=0)
        with pytest.raises(ValueError):
            cumulate_horizons(daily, h=6)

    def test_insufficient_daily_raises(self):
        with pytest.raises(ValueError):
            cumulate_horizons([0.1, 0.2], h=5)


# ── Consistency: path compound matches daily compounding ───────────────


class TestPathConsistency:
    def _path(self) -> ReturnPath:
        prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 10, 11),
                _bar("2026-01-06T00:00:00+00:00", 11, 12),
                _bar("2026-01-07T00:00:00+00:00", 12, 11),
                _bar("2026-01-08T00:00:00+00:00", 11, 13),
                _bar("2026-01-09T00:00:00+00:00", 13, 14),
            ]
        )
        return compute_return_path(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
        )

    def test_r1_matches_daily_compound(self):
        path = self._path()
        daily = [path.d1_return, path.d2_return, path.d3_return, path.d4_return, path.d5_return]
        r1 = cumulate_horizons(daily, h=1)
        assert r1 == pytest.approx(path.d1_return, abs=1e-12)

    def test_r3_matches_daily_compound(self):
        path = self._path()
        daily = [path.d1_return, path.d2_return, path.d3_return, path.d4_return, path.d5_return]
        r3 = cumulate_horizons(daily, h=3)
        # exit_3d_price / entry_price - 1 must equal the compound
        assert r3 == pytest.approx(path.exit_3d_price / path.entry_price - 1.0, abs=1e-12)

    def test_r5_matches_daily_compound(self):
        path = self._path()
        daily = [path.d1_return, path.d2_return, path.d3_return, path.d4_return, path.d5_return]
        r5 = cumulate_horizons(daily, h=5)
        assert r5 == pytest.approx(path.exit_5d_price / path.entry_price - 1.0, abs=1e-12)


# ── Market excess computation ──────────────────────────────────────────


class TestMarketExcess:
    def test_excess_is_stock_minus_market_compound(self):
        stock_prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 10, 11),
                _bar("2026-01-06T00:00:00+00:00", 11, 12),
                _bar("2026-01-07T00:00:00+00:00", 12, 11),
                _bar("2026-01-08T00:00:00+00:00", 11, 13),
                _bar("2026-01-09T00:00:00+00:00", 13, 14),
            ]
        )
        # Market doubles over the horizon: 100 -> 200 (cum 5d = 1.0)
        market_prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 100, 110),
                _bar("2026-01-06T00:00:00+00:00", 110, 121),
                _bar("2026-01-07T00:00:00+00:00", 121, 133.1),
                _bar("2026-01-08T00:00:00+00:00", 133.1, 146.41),
                _bar("2026-01-09T00:00:00+00:00", 146.41, 161.051),
            ]
        )
        excess = compute_market_excess_path(
            stock_id="000001.SZ",
            market_id="SHA",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=stock_prices,
            market_prices_df=market_prices,
        )
        assert isinstance(excess, MarketExcessPath)
        assert excess.stock_id == "000001.SZ"
        assert excess.market_id == "SHA"

        stock_path = compute_return_path(
            stock_id="000001.SZ",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=stock_prices,
        )
        market_path = compute_return_path(
            stock_id="SHA",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=market_prices,
        )

        # Per-day excess = stock daily - market daily
        assert excess.d1_excess == pytest.approx(
            stock_path.d1_return - market_path.d1_return, abs=1e-12
        )
        # Cumulative excess = stock compound - market compound
        stock_daily = [
            stock_path.d1_return,
            stock_path.d2_return,
            stock_path.d3_return,
            stock_path.d4_return,
            stock_path.d5_return,
        ]
        market_daily = [
            market_path.d1_return,
            market_path.d2_return,
            market_path.d3_return,
            market_path.d4_return,
            market_path.d5_return,
        ]
        assert excess.cum_excess_1d == pytest.approx(
            cumulate_horizons(stock_daily, h=1) - cumulate_horizons(market_daily, h=1),
            abs=1e-12,
        )
        assert excess.cum_excess_3d == pytest.approx(
            cumulate_horizons(stock_daily, h=3) - cumulate_horizons(market_daily, h=3),
            abs=1e-12,
        )
        assert excess.cum_excess_5d == pytest.approx(
            cumulate_horizons(stock_daily, h=5) - cumulate_horizons(market_daily, h=5),
            abs=1e-12,
        )

    def test_excess_zero_when_stock_equals_market(self):
        prices = _price_df(
            [
                _bar("2026-01-05T00:00:00+00:00", 10, 11),
                _bar("2026-01-06T00:00:00+00:00", 11, 12),
                _bar("2026-01-07T00:00:00+00:00", 12, 11),
                _bar("2026-01-08T00:00:00+00:00", 11, 13),
                _bar("2026-01-09T00:00:00+00:00", 13, 14),
            ]
        )
        excess = compute_market_excess_path(
            stock_id="000001.SZ",
            market_id="SHA",
            entry_date=_utc("2026-01-05T00:00:00+00:00"),
            prices_df=prices,
            market_prices_df=prices,
        )
        assert excess.cum_excess_1d == pytest.approx(0.0, abs=1e-12)
        assert excess.cum_excess_3d == pytest.approx(0.0, abs=1e-12)
        assert excess.cum_excess_5d == pytest.approx(0.0, abs=1e-12)
