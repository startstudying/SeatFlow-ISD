"""PIT seat history feature computation (Task 10).

This module computes point-in-time (PIT) statistics for a seat based on
its LHB participations and mature return labels. The key PIT rules are:

- Only participations whose ``available_time <= at_time`` are used.
- Only labels whose ``mature_time <= at_time`` are used.
- Aggregation windows (5d, 20d) count calendar days back from ``at_time``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from seatflow.dataset.event import SeatParticipation
from seatflow.domain.time import assert_aware


# --- SeatHistory dataclass -------------------------------------------


@dataclass(frozen=True)
class SeatHistory:
    """PIT snapshot of a seat's activity statistics.

    Attributes
    ----------
    seat_id:
        The resolved seat identifier.
    snapshot_time:
        The ``at_time`` at which this snapshot was computed.
    lhb_count_5d:
        Number of LHB participations in the 5-calendar-day window.
    lhb_count_20d:
        Number of LHB participations in the 20-calendar-day window.
    buy_amount_5d:
        Total buy amount over the 5-day window.
    sell_amount_5d:
        Total sell amount over the 5-day window.
    net_amount_5d:
        Net amount (buy - sell) over the 5-day window.
    mature_return_mean:
        Mean of mature returns for labels available at ``snapshot_time``.
        Zero if no mature labels.
    win_rate:
        Fraction of mature returns > 0. Zero if no mature labels.
    event_reliability:
        Confidence-weighted reliability score in [0, 1].
    """

    seat_id: str
    snapshot_time: datetime
    lhb_count_5d: int
    lhb_count_20d: int
    buy_amount_5d: float
    sell_amount_5d: float
    net_amount_5d: float
    mature_return_mean: float
    win_rate: float
    event_reliability: float

    def __post_init__(self) -> None:
        assert_aware(self.snapshot_time, "snapshot_time")


# --- compute_seat_history --------------------------------------------


def compute_seat_history(
    seat_id: str,
    at_time: datetime,
    participations: list[SeatParticipation],
    mature_labels: list[Any],
) -> SeatHistory:
    """Compute a PIT seat history snapshot.

    Parameters
    ----------
    seat_id:
        The seat to compute history for.
    at_time:
        The PIT cutoff. Only participations with ``available_time <=
        at_time`` and labels with ``mature_time <= at_time`` are used.
    participations:
        All participations for this seat (PIT filtering applied here).
    mature_labels:
        A list of label-like objects. Each must expose either attributes
        ``mature_time`` / ``mature_return`` or mapping keys of the same
        name.

    Returns
    -------
    SeatHistory
        The computed snapshot.
    """
    assert_aware(at_time, "at_time")

    # --- PIT participation filtering ---------------------------------
    pit_parts = [
        p
        for p in participations
        if p.seat_id == seat_id and p.available_time <= at_time
    ]

    window_5d_start = at_time - timedelta(days=5)
    window_20d_start = at_time - timedelta(days=20)

    parts_5d = [p for p in pit_parts if p.available_time >= window_5d_start]
    parts_20d = [p for p in pit_parts if p.available_time >= window_20d_start]

    buy_amount_5d = sum(p.buy_amount for p in parts_5d)
    sell_amount_5d = sum(p.sell_amount for p in parts_5d)
    net_amount_5d = buy_amount_5d - sell_amount_5d

    # --- Mature label filtering --------------------------------------
    mature_returns: list[float] = []
    for label in mature_labels:
        m_time, m_ret = _extract_label(label)
        if m_time is None:
            continue
        assert_aware(m_time, "label.mature_time")
        if m_time <= at_time:
            mature_returns.append(m_ret)

    if mature_returns:
        mature_return_mean = sum(mature_returns) / len(mature_returns)
        wins = sum(1 for r in mature_returns if r > 0.0)
        win_rate = wins / len(mature_returns)
    else:
        mature_return_mean = 0.0
        win_rate = 0.0

    # --- Event reliability -------------------------------------------
    # Reliability is the mean identity_confidence of PIT participations,
    # or 0.0 if there are none.
    if pit_parts:
        event_reliability = sum(p.identity_confidence for p in pit_parts) / len(pit_parts)
    else:
        event_reliability = 0.0

    return SeatHistory(
        seat_id=seat_id,
        snapshot_time=at_time,
        lhb_count_5d=len(parts_5d),
        lhb_count_20d=len(parts_20d),
        buy_amount_5d=buy_amount_5d,
        sell_amount_5d=sell_amount_5d,
        net_amount_5d=net_amount_5d,
        mature_return_mean=mature_return_mean,
        win_rate=win_rate,
        event_reliability=event_reliability,
    )


def _extract_label(label: Any) -> tuple[datetime | None, float]:
    """Extract ``mature_time`` and ``mature_return`` from a label-like
    object (supports both attribute and mapping access)."""
    # Mapping access (dict-like).
    if isinstance(label, dict) or hasattr(label, "get"):
        try:
            m_time = label.get("mature_time")  # type: ignore[union-attr]
            m_ret = label.get("mature_return", 0.0)  # type: ignore[union-attr]
            return m_time, float(m_ret)
        except (TypeError, ValueError):
            pass
    # Attribute access.
    m_time = getattr(label, "mature_time", None)
    m_ret = getattr(label, "mature_return", 0.0)
    return m_time, float(m_ret)
