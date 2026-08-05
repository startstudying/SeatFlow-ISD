"""Tradeability classification for prediction samples (Task 12).

A prediction sample is only *backtest-eligible* if the stock was actually
tradable at the entry point and throughout the 5-day horizon. A label
computed on a stock that was suspended, locked at a one-price limit, or
delisted mid-horizon would be unrealizable in a backtest and must be
flagged so the sealer can reject it.

The classifier inspects the OHLC bar for the entry date and the subsequent
bars and returns one of the :class:`TradeabilityState` values. The checks
are applied in priority order:

1. ``MISSING_ENTRY_PRICE`` — no bar exists for the entry date.
2. ``SUSPENDED_AT_ENTRY`` — the entry-day bar has ``volume == 0`` (or an
   explicit ``suspended`` flag).
3. ``ONE_PRICE_LIMIT_UP`` / ``ONE_PRICE_LIMIT_DOWN`` — the entry-day bar is
   a one-price bar (open == close == high == low) with a ``limit_up`` /
   ``limit_down`` marker and non-zero volume.
4. ``DELISTED_DURING_HORIZON`` — fewer than 5 bars remain after the entry
   date, or any subsequent bar carries a ``delisted`` marker.
5. ``TRADABLE`` — otherwise.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Sequence

from seatflow.domain.time import assert_aware


# ── Enum ───────────────────────────────────────────────────────────────


class TradeabilityState(str, Enum):
    """Whether a stock was tradable at entry and through the horizon."""

    TRADABLE = "TRADABLE"
    SUSPENDED_AT_ENTRY = "SUSPENDED_AT_ENTRY"
    ONE_PRICE_LIMIT_UP = "ONE_PRICE_LIMIT_UP"
    ONE_PRICE_LIMIT_DOWN = "ONE_PRICE_LIMIT_DOWN"
    MISSING_ENTRY_PRICE = "MISSING_ENTRY_PRICE"
    DELISTED_DURING_HORIZON = "DELISTED_DURING_HORIZON"


# ── Internal helpers ───────────────────────────────────────────────────


_HORIZON_DAYS = 5


def _find_entry_index(
    prices: Sequence[dict],
    entry_date: datetime,
) -> int:
    """Return the index of the bar matching *entry_date*, or ``-1``."""
    for i, bar in enumerate(prices):
        if bar["date"] == entry_date:
            return i
    return -1


def _is_one_price(bar: dict) -> bool:
    o = float(bar.get("open", 0.0))
    c = float(bar.get("close", 0.0))
    h = float(bar.get("high", o))
    low = float(bar.get("low", o))
    return o == c == h == low


def _is_suspended(bar: dict) -> bool:
    if bar.get("suspended"):
        return True
    # volume == 0 with a one-price bar is treated as a suspension.
    volume = float(bar.get("volume", 0.0))
    return volume == 0.0 and _is_one_price(bar)


def _is_delisted(bar: dict) -> bool:
    return bool(bar.get("delisted", False))


# ── Public API ─────────────────────────────────────────────────────────


def classify_tradeability(
    *,
    stock_id: str,
    entry_date: datetime,
    prices_df: Sequence[dict],
) -> TradeabilityState:
    """Classify the tradeability of *stock_id* at *entry_date*.

    Parameters
    ----------
    stock_id:
        Canonical stock identifier (carried for logging/auditing only).
    entry_date:
        Timezone-aware UTC datetime of the entry bar.
    prices_df:
        Ordered sequence of OHLC bar dicts. Each bar must have at least
        ``date`` (tz-aware UTC), ``open``, ``close``, ``high``, ``low``,
        and ``volume``. Optional flags ``suspended``, ``limit_up``,
        ``limit_down`` and ``delisted`` refine the classification.

    Returns
    -------
    TradeabilityState
    """
    del stock_id  # not used in the decision; kept for API symmetry
    assert_aware(entry_date, "entry_date")

    if not prices_df:
        return TradeabilityState.MISSING_ENTRY_PRICE

    idx = _find_entry_index(prices_df, entry_date)
    if idx < 0:
        return TradeabilityState.MISSING_ENTRY_PRICE

    entry_bar = prices_df[idx]

    # 1. Suspension on the entry day takes precedence over a later
    #    delisting: there is no entry trade at all.
    if _is_suspended(entry_bar):
        return TradeabilityState.SUSPENDED_AT_ENTRY

    # 2. One-price limit bar (non-zero volume so it is not a suspension).
    if _is_one_price(entry_bar):
        volume = float(entry_bar.get("volume", 0.0))
        if volume > 0.0:
            if entry_bar.get("limit_down"):
                return TradeabilityState.ONE_PRICE_LIMIT_DOWN
            if entry_bar.get("limit_up"):
                return TradeabilityState.ONE_PRICE_LIMIT_UP
            # A one-price bar with volume but no limit marker: if it
            # opened low and the close equals the low, treat as limit-down;
            # otherwise limit-up. This makes the classifier robust to
            # providers that omit the explicit flag.
            open_ = float(entry_bar.get("open", 0.0))
            low = float(entry_bar.get("low", open_))
            if open_ == low:
                return TradeabilityState.ONE_PRICE_LIMIT_DOWN
            return TradeabilityState.ONE_PRICE_LIMIT_UP

    # 3. Delisting during the horizon: fewer than 5 bars from the entry
    #    onward, or any subsequent bar explicitly marked delisted.
    tail = prices_df[idx:]
    if len(tail) < _HORIZON_DAYS:
        return TradeabilityState.DELISTED_DURING_HORIZON
    for bar in tail[:_HORIZON_DAYS]:
        if _is_delisted(bar):
            return TradeabilityState.DELISTED_DURING_HORIZON

    return TradeabilityState.TRADABLE
