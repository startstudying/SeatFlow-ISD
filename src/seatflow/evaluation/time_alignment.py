"""Utilities for validating temporal alignment between data availability and trade execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TradeTimeAlignmentAudit:
    status: str
    message: str
    checked_records: int
    evidence_gaps: tuple[str, ...]


def _coerce_datetime(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label} must be ISO datetime string, got {value!r}") from error
    raise ValueError(f"{label} must be datetime or ISO datetime string, got {type(value).__name__}")


def _assert_timezone_compatible(a: datetime, b: datetime, *, context: str) -> None:
    if (a.tzinfo is None) != (b.tzinfo is None):
        raise ValueError(f"{context} has mixed timezone awareness")
    if a.tzinfo is not None and b.tzinfo is not None:
        if a.utcoffset() != b.utcoffset():
            raise ValueError(f"{context} has incompatible timezone offsets")


def assert_trade_time_after_data_available(
    trade_time: datetime | str,
    data_available_time: datetime | str,
    *,
    context: str = "trade row",
) -> None:
    """Raise if the trade time is earlier than data availability time."""
    parsed_trade_time = _coerce_datetime(trade_time, "trade_time")
    parsed_available_time = _coerce_datetime(data_available_time, "data_available_time")
    _assert_timezone_compatible(
        parsed_trade_time, parsed_available_time, context=f"{context} timezone"
    )
    if parsed_trade_time < parsed_available_time:
        raise ValueError(
            f"{context}: trade_time cannot precede available_time "
            f"({parsed_trade_time.isoformat()} < {parsed_available_time.isoformat()})"
        )


def evaluate_trade_time_alignment(
    records: Sequence[object],
    *,
    trade_time_key: str = "trade_time",
    available_time_key: str = "data_available_time",
    context_key: str = "context",
) -> TradeTimeAlignmentAudit:
    """
    Evaluate a batch of records and return an evidence status.

    Missing required fields are reported as insufficient evidence because the row
    cannot be interpreted deterministically.
    """
    gaps: list[str] = []
    checked = 0
    for index, record in enumerate(records, start=1):
        if isinstance(record, dict):
            trade_time = record.get(trade_time_key)
            available_time = record.get(available_time_key)
            context = str(record.get(context_key, f"record #{index}"))
        else:
            trade_time = getattr(record, trade_time_key, None)
            available_time = getattr(record, available_time_key, None)
            context = str(getattr(record, context_key, f"record #{index}"))
        if trade_time is None or available_time is None:
            gaps.append(f"{context}: missing {trade_time_key} or {available_time_key}")
            continue
        checked += 1
        try:
            assert_trade_time_after_data_available(
                trade_time=trade_time,
                data_available_time=available_time,
                context=context,
            )
        except ValueError as error:
            raise ValueError(
                f"{context}: {error}"
            ) from None
    if not records:
        return TradeTimeAlignmentAudit(
            status="insufficient_evidence",
            message="no records were provided",
            checked_records=0,
            evidence_gaps=("no_records",),
        )
    if gaps:
        return TradeTimeAlignmentAudit(
            status="insufficient_evidence",
            message="trade-time evidence is incomplete",
            checked_records=checked,
            evidence_gaps=tuple(gaps),
        )
    return TradeTimeAlignmentAudit(
        status="confirmed",
        message="all trade times are after data availability",
        checked_records=checked,
        evidence_gaps=(),
    )


__all__ = [
    "TradeTimeAlignmentAudit",
    "assert_trade_time_after_data_available",
    "evaluate_trade_time_alignment",
]
