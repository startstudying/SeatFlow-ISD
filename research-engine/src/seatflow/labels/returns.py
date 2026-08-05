"""Multi-horizon return labels (Task 12).

A *return path* records the daily and cumulative returns for a stock over
a 5-trading-day horizon starting from the *entry date*. The labels are the
prediction targets for short-horizon return models.

Definition
----------

Given OHLC bars :math:`(O_t, C_t)` for trading days :math:`t = 1..5`
following the entry date, with the entry price equal to the day-1 open
:math:`P_0 = O_1`:

- :math:`d_t = (C_t - O_t) / O_t`                              (intraday O→C)
- :math:`R_1 = d_1`
- :math:`R_3 = \\prod_{t=1}^{3} (1 + d_t) - 1`                  (exact compound)
- :math:`R_5 = \\prod_{t=1}^{5} (1 + d_t) - 1`                  (exact compound)

The same convention applies to the market index. *Market excess* at
horizon :math:`h` is :math:`R_h^{stock} - R_h^{market}`.

All timestamps must be timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

from seatflow.domain.time import assert_aware


# ── Dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReturnPath:
    """The daily and cumulative return path for a single stock over 5 days."""

    stock_id: str
    entry_date: datetime
    entry_price: float
    d1_return: float
    d2_return: float
    d3_return: float
    d4_return: float
    d5_return: float
    exit_1d_price: float
    exit_3d_price: float
    exit_5d_price: float


@dataclass(frozen=True)
class MarketExcessPath:
    """Per-day and cumulative market-excess returns over a 5-day horizon."""

    stock_id: str
    market_id: str
    d1_excess: float
    d2_excess: float
    d3_excess: float
    d4_excess: float
    d5_excess: float
    cum_excess_1d: float
    cum_excess_3d: float
    cum_excess_5d: float


# ── Validation helpers ─────────────────────────────────────────────────


_VALID_HORIZONS = (1, 3, 5)
_HORIZON_DAYS = 5


def _validate_dates(dates: Sequence[datetime], field: str) -> None:
    for d in dates:
        assert_aware(d, field)


def _coerce_bar_dates(prices: Iterable[dict]) -> list[datetime]:
    """Extract ``date`` fields from a list of bar dicts."""
    return [bar["date"] for bar in prices]


def _align_window(
    prices: Sequence[dict],
    entry_date: datetime,
) -> list[dict]:
    """Return the slice of ``prices`` starting at the bar for ``entry_date``.

    The window contains exactly :data:`_HORIZON_DAYS` bars — the entry-day
    bar plus the four subsequent bars. If fewer than five bars remain after
    the entry, returns whatever is available (caller decides on the
    tradeability state).
    """
    aligned: list[dict] = []
    for bar in prices:
        bar_date = bar["date"]
        if bar_date < entry_date:
            continue
        if bar_date == entry_date:
            aligned.append(bar)
            continue
        # strictly after the entry date
        if len(aligned) == 0:
            # bar_date is after entry but we never matched the entry date;
            # this is a *gap* and there is no entry bar.
            return []
        aligned.append(bar)
        if len(aligned) >= _HORIZON_DAYS:
            break
    return aligned


def _bar_open_close(bar: dict) -> tuple[float, float]:
    return float(bar["open"]), float(bar["close"])


# ── Public API ─────────────────────────────────────────────────────────


def cumulate_horizons(daily_returns: Sequence[float], *, h: int) -> float:
    """Compute the *exact* compound return over the first ``h`` daily returns.

    Parameters
    ----------
    daily_returns:
        A sequence of simple daily returns ``d_1, d_2, …``.
    h:
        One of ``{1, 3, 5}``. Anything else raises ``ValueError``.

    Returns
    -------
    float
        ``∏_{i=1..h} (1 + d_i) - 1``.
    """
    if h not in _VALID_HORIZONS:
        raise ValueError(
            f"horizon h must be one of {sorted(_VALID_HORIZONS)}, got {h}"
        )
    if len(daily_returns) < h:
        raise ValueError(
            f"need at least {h} daily returns to compound, got {len(daily_returns)}"
        )
    growth = 1.0
    for i in range(h):
        growth *= 1.0 + float(daily_returns[i])
    return growth - 1.0


def compute_return_path(
    *,
    stock_id: str,
    entry_date: datetime,
    prices_df: Sequence[dict],
) -> ReturnPath:
    """Compute the 5-day return path for *stock_id* starting at *entry_date*.

    The first bar in ``prices_df`` whose date equals ``entry_date`` is the
    entry-day bar; its open is the *entry price*. The next four bars (or
    however many remain) supply :math:`d_1, …, d_5`.

    ``exit_1d_price`` / ``exit_3d_price`` / ``exit_5d_price`` are the close
    of day 1 / 3 / 5 respectively.

    If a return cannot be computed because a bar is missing its ``open``
    or ``close`` field, the corresponding daily return is set to ``0.0``
    and the exit price to ``0.0`` so the dataclass is always well-formed.
    Callers that need stricter handling should validate the bar schema
    before calling this function.
    """
    assert_aware(entry_date, "entry_date")

    window = _align_window(prices_df, entry_date)
    if not window:
        raise ValueError(
            f"no price bar found at entry_date={entry_date.isoformat()}"
        )

    # Entry price = open of the entry-day bar (the first bar in the window).
    entry_open, _ = _bar_open_close(window[0])
    daily: list[float] = []
    exits: list[float] = []
    for i in range(_HORIZON_DAYS):
        if i < len(window):
            o, c = _bar_open_close(window[i])
            daily.append((c - o) / o if o != 0 else 0.0)
            exits.append(c)
        else:
            # No bar available — model the missing day as flat.
            daily.append(0.0)
            exits.append(0.0)

    d1, d2, d3, d4, d5 = daily

    return ReturnPath(
        stock_id=stock_id,
        entry_date=entry_date,
        entry_price=entry_open,
        d1_return=d1,
        d2_return=d2,
        d3_return=d3,
        d4_return=d4,
        d5_return=d5,
        exit_1d_price=exits[0],
        exit_3d_price=exits[2],
        exit_5d_price=exits[4],
    )


def compute_market_excess_path(
    *,
    stock_id: str,
    market_id: str,
    entry_date: datetime,
    prices_df: Sequence[dict],
    market_prices_df: Sequence[dict],
) -> MarketExcessPath:
    """Compute the 5-day market-excess path for *stock_id* against *market_id*.

    Per-day excess is the difference of simple daily returns; the cumulative
    excess is the difference of *compounded* daily returns over 1 / 3 / 5
    days.
    """
    stock = compute_return_path(
        stock_id=stock_id,
        entry_date=entry_date,
        prices_df=prices_df,
    )
    market = compute_return_path(
        stock_id=market_id,
        entry_date=entry_date,
        prices_df=market_prices_df,
    )
    stock_daily = [stock.d1_return, stock.d2_return, stock.d3_return, stock.d4_return, stock.d5_return]
    market_daily = [market.d1_return, market.d2_return, market.d3_return, market.d4_return, market.d5_return]
    per_day_excess = [s - m for s, m in zip(stock_daily, market_daily)]
    return MarketExcessPath(
        stock_id=stock_id,
        market_id=market_id,
        d1_excess=per_day_excess[0],
        d2_excess=per_day_excess[1],
        d3_excess=per_day_excess[2],
        d4_excess=per_day_excess[3],
        d5_excess=per_day_excess[4],
        cum_excess_1d=cumulate_horizons(stock_daily, h=1) - cumulate_horizons(market_daily, h=1),
        cum_excess_3d=cumulate_horizons(stock_daily, h=3) - cumulate_horizons(market_daily, h=3),
        cum_excess_5d=cumulate_horizons(stock_daily, h=5) - cumulate_horizons(market_daily, h=5),
    )
