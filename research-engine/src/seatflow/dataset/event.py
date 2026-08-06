"""Event extraction and point-in-time (PIT) availability (Task 8).

Dragon-Tiger-List (LHB) events are the primary attention signal in
SeatFlow. This module converts Bronze records (raw provider payloads)
into immutable :class:`LHBEvent` and :class:`SeatParticipation` objects.

PIT contract
------------
- ``available_time`` is when the system could first observe the event and
  must satisfy ``available_time >= trade_date``. Pre-announcement is
  forbidden for LHB events (the list is published after the close).
- Seat identities are resolved at ``available_time`` (not ``trade_date``)
  so that a seat renamed between the trade date and the publication date
  is resolved to the identity that was canonical *when the record became
  observable*. This prevents silently rewriting history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from seatflow.domain.time import assert_aware
from seatflow.identity.seat_identity import SeatIdentityRegistry, SeatIdentityResolution
from seatflow.ingestion.models import BronzeRecord, SourceConfidence


# ── Confidence mapping ─────────────────────────────────────────────────


_SOURCE_CONFIDENCE_TO_FLOAT: dict[SourceConfidence, float] = {
    SourceConfidence.HIGH: 1.0,
    SourceConfidence.MEDIUM: 0.7,
    SourceConfidence.LOW: 0.4,
    SourceConfidence.DEGRADED: 0.2,
}


def _confidence_from_source(source_confidence: SourceConfidence) -> float:
    """Map a provider confidence level to a numeric event confidence."""
    return _SOURCE_CONFIDENCE_TO_FLOAT.get(source_confidence, 0.0)


# ── Enums ──────────────────────────────────────────────────────────────


class SeatSide(str, Enum):
    """Side of a seat's participation in an LHB event."""

    BUY = "BUY"
    SELL = "SELL"


# ── Dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LHBEvent:
    """A single Dragon-Tiger-List event for one stock on one trade date.

    Attributes
    ----------
    event_id:
        Stable string identifier derived from source lineage and record id.
    stock_id:
        Canonical stock code (e.g. ``"000001.SZ"``).
    trade_date:
        The trade date the LHB event refers to (midnight UTC, tz-aware).
    available_time:
        When the event first became observable by the system. Enforced to
        be ``>= trade_date`` to prevent look-ahead bias.
    reason_codes:
        Reasons the stock appeared on the LHB (e.g. price deviation).
    total_buy:
        Total buy amount across all participating seats (CNY).
    total_sell:
        Total sell amount across all participating seats (CNY).
    net_amount:
        ``total_buy - total_sell``.
    turnover_rate:
        Turnover rate of the stock on the trade date (percent).
    event_confidence:
        Confidence in the event derived from the source confidence.
    source_lineage:
        Name of the provider that produced the underlying Bronze record.
    """

    event_id: str
    stock_id: str
    trade_date: datetime
    available_time: datetime
    reason_codes: list[str] = field(default_factory=list)
    total_buy: float = 0.0
    total_sell: float = 0.0
    net_amount: float = 0.0
    turnover_rate: float = 0.0
    event_confidence: float = 1.0
    source_lineage: str = ""


@dataclass(frozen=True)
class SeatParticipation:
    """A single seat's participation in one LHB event.

    Attributes
    ----------
    event_id:
        The :attr:`LHBEvent.event_id` this participation belongs to.
    seat_id:
        Canonical seat identifier resolved at ``available_time``.
    side:
        :attr:`SeatSide.BUY` or :attr:`SeatSide.SELL`.
    rank:
        Rank of the seat within its side (1-based).
    buy_amount:
        Buy amount attributed to the seat (CNY).
    sell_amount:
        Sell amount attributed to the seat (CNY).
    net_amount:
        ``buy_amount - sell_amount``.
    amount_share:
        The seat's share of the total amount on its side
        (``buy_amount / total_buy`` for BUY, ``sell_amount / total_sell``
        for SELL).
    available_time:
        The event's ``available_time`` — used for PIT resolution.
    identity_confidence:
        Confidence carried from the :class:`SeatIdentityResolution`.
    """

    event_id: str
    seat_id: str
    side: SeatSide
    rank: int
    buy_amount: float
    sell_amount: float
    net_amount: float
    amount_share: float
    available_time: datetime
    identity_confidence: float


# ── ID generation ──────────────────────────────────────────────────────


def _event_id(source: str, source_record_id: str) -> str:
    """Derive a stable event_id from source lineage and the bronze record id.

    The ID is a string (never a bare numeric) so it cannot leak into
    numeric features.
    """
    raw = f"{source}|{source_record_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"lhb-{digest}"


# ── Payload helpers ────────────────────────────────────────────────────


def _reason_codes(payload: dict[str, Any]) -> list[str]:
    """Extract reason codes from the raw payload.

    The Tushare ``reason`` field is a comma- or semicolon-joined string.
    Empty entries are dropped.
    """
    raw = payload.get("reason", "")
    if isinstance(raw, list):
        return [str(r).strip() for r in raw if str(r).strip()]
    s = str(raw)
    # Split on both ',' and ';' to be robust across providers.
    parts = s.replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def _safe_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = payload.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Extraction ─────────────────────────────────────────────────────────


def _assert_aware_and_pit(trade_date: datetime, available_time: datetime) -> None:
    """Enforce tz-awareness and the PIT ordering for LHB events."""
    assert_aware(trade_date, "trade_date")
    assert_aware(available_time, "available_time")
    if available_time < trade_date:
        raise ValueError(
            f"available_time ({available_time}) must be >= trade_date "
            f"({trade_date}) for LHB events — pre-announcement is forbidden"
        )


def extract_events_from_bronze(
    bronze_records: list[BronzeRecord],
    seat_registry: SeatIdentityRegistry,
) -> list[LHBEvent]:
    """Convert Bronze records into :class:`LHBEvent` objects.

    Each Bronze record's ``raw_payload`` is expected to carry the LHB
    summary fields (``ts_code``, ``trade_date``, ``ann_date``, ``reason``,
    ``total_buy``, ``total_sell``, ``net_amount``, ``turnover_rate``).
    ``trade_date`` is used as the ``event_time`` and ``ann_date`` as the
    ``available_time``.

    The PIT ordering ``available_time >= trade_date`` is enforced; a
    record violating it raises ``ValueError``.

    The ``seat_registry`` is accepted for API symmetry (and so callers
    can extract events and participations with the same arguments) but is
    not needed to build the event itself — seat resolution happens in
    :func:`extract_participations_from_bronze`.
    """
    del seat_registry  # not needed for event-level extraction
    events: list[LHBEvent] = []
    for rec in bronze_records:
        payload = rec.raw_payload
        stock_id = str(payload.get("ts_code", ""))
        trade_date = rec.event_time
        available_time = rec.available_time
        _assert_aware_and_pit(trade_date, available_time)

        total_buy = _safe_float(payload, "total_buy")
        total_sell = _safe_float(payload, "total_sell")
        net_amount = _safe_float(payload, "net_amount", total_buy - total_sell)
        turnover_rate = _safe_float(payload, "turnover_rate")

        events.append(
            LHBEvent(
                event_id=_event_id(rec.source, rec.source_record_id),
                stock_id=stock_id,
                trade_date=trade_date,
                available_time=available_time,
                reason_codes=_reason_codes(payload),
                total_buy=total_buy,
                total_sell=total_sell,
                net_amount=net_amount,
                turnover_rate=turnover_rate,
                event_confidence=_confidence_from_source(rec.source_confidence),
                source_lineage=rec.source,
            )
        )
    return events


def extract_participations_from_bronze(
    bronze_records: list[BronzeRecord],
    seat_registry: SeatIdentityRegistry,
) -> list[SeatParticipation]:
    """Extract seat participations from Bronze records, resolving seat
    identities at each record's ``available_time``.

    Each Bronze record's ``raw_payload`` is expected to carry two lists:

    - ``seats_buy``: list of ``{name, rank, buy_amount, sell_amount}```
    - ``seats_sell``: list of ``{name, rank, buy_amount, sell_amount}``

    The seat ``name`` is resolved against ``seat_registry`` at the
    record's ``available_time`` (point-in-time resolution). A seat that is
    known but outside its validity window at ``available_time`` raises
    ``KeyError`` (propagated from the registry). An unregistered seat
    yields an UNKNOWN identity with ``identity_confidence == 0.0``.

    ``amount_share`` is the seat's share of the *side total*: for BUY
    seats it is ``buy_amount / total_buy`` and for SELL seats it is
    ``sell_amount / total_sell``.
    """
    participations: list[SeatParticipation] = []
    for rec in bronze_records:
        payload = rec.raw_payload
        trade_date = rec.event_time
        available_time = rec.available_time
        _assert_aware_and_pit(trade_date, available_time)

        event_id = _event_id(rec.source, rec.source_record_id)
        total_buy = _safe_float(payload, "total_buy")
        total_sell = _safe_float(payload, "total_sell")

        buy_seats = payload.get("seats_buy", []) or []
        sell_seats = payload.get("seats_sell", []) or []

        for seat_row in buy_seats:
            participations.append(
                _build_participation(
                    seat_row,
                    side=SeatSide.BUY,
                    event_id=event_id,
                    available_time=available_time,
                    side_total=total_buy,
                    seat_registry=seat_registry,
                )
            )
        for seat_row in sell_seats:
            participations.append(
                _build_participation(
                    seat_row,
                    side=SeatSide.SELL,
                    event_id=event_id,
                    available_time=available_time,
                    side_total=total_sell,
                    seat_registry=seat_registry,
                )
            )
    return participations


def _build_participation(
    seat_row: dict[str, Any],
    *,
    side: SeatSide,
    event_id: str,
    available_time: datetime,
    side_total: float,
    seat_registry: SeatIdentityRegistry,
) -> SeatParticipation:
    """Build a single SeatParticipation, resolving the seat at available_time."""
    raw_name = str(seat_row.get("name", ""))
    rank = int(seat_row.get("rank", 0))
    buy_amount = _safe_float(seat_row, "buy_amount")
    sell_amount = _safe_float(seat_row, "sell_amount")

    # PIT resolution at available_time (not trade_date). A known-but-
    # out-of-window seat raises KeyError here, which we let propagate.
    resolution: SeatIdentityResolution = seat_registry.resolve_seat(
        raw_name, available_time
    )

    if side == SeatSide.BUY:
        share = (buy_amount / side_total) if side_total > 0 else 0.0
    else:
        share = (sell_amount / side_total) if side_total > 0 else 0.0

    return SeatParticipation(
        event_id=event_id,
        seat_id=resolution.seat_id,
        side=side,
        rank=rank,
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        net_amount=buy_amount - sell_amount,
        amount_share=share,
        available_time=available_time,
        identity_confidence=resolution.identity_confidence,
    )
