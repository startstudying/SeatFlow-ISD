"""Tradeability source for SeatFlow-ISD V1.

Records per-(stock, date) tradeability signals:

- ``suspend`` — whether the stock is suspended that day.
- ``limit_up`` / ``limit_down`` — whether the day is a one-price
  limit-up / limit-down bar.
- ``one_price`` — whether open == close == high == low (one-price
  board).

The expected on-disk layout follows the standard
``date=YYYY-MM-DD`` + ``data.json`` + ``manifest.json`` convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from seatflow.dataset.source_snapshots import (
    SnapshotRecord,
    discover_and_load_records,
)
from seatflow.domain.time import assert_aware


@dataclass(frozen=True)
class TradeabilityRow:
    """One daily tradeability signal row."""

    stock_id: str
    trade_date: datetime
    suspended: bool
    limit_up: bool
    limit_down: bool
    one_price: bool
    available_time: datetime

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")
        if self.limit_up and self.limit_down:
            raise ValueError(
                f"stock {self.stock_id} on {self.trade_date} cannot be both "
                f"limit_up and limit_down"
            )


@dataclass(frozen=True)
class TradeabilitySource:
    """Verified metadata for the tradeability source."""

    data_root: Path
    manifest_fingerprint: str
    date_count: int
    row_count: int
    verified: bool


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert_aware(parsed, "datetime")
    return parsed


def _row_to_tradeability(row: Mapping[str, Any]) -> TradeabilityRow:
    return TradeabilityRow(
        stock_id=str(row["stock_id"]),
        trade_date=_parse_dt(row["trade_date"]),
        suspended=bool(row.get("suspended", False)),
        limit_up=bool(row.get("limit_up", False)),
        limit_down=bool(row.get("limit_down", False)),
        one_price=bool(row.get("one_price", False)),
        available_time=_parse_dt(row["available_time"]),
    )


def discover_tradeability_records(
    source_root: Path,
) -> list[tuple[SnapshotRecord, list[dict[str, Any]]]]:
    """Discover and load every verified tradeability snapshot."""
    dataset_dir = source_root / "tradeability"
    if not dataset_dir.exists():
        return []
    return discover_and_load_records(dataset_dir.parent, Path(dataset_dir.name))


def load_tradeability(
    source_root: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[TradeabilityRow], TradeabilitySource]:
    """Load verified tradeability rows."""
    pairs = discover_tradeability_records(source_root)
    if not pairs:
        raise FileNotFoundError(
            f"no verified snapshots for tradeability at {source_root}"
        )
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    rows: list[TradeabilityRow] = []
    for record, loaded in pairs:
        if start is not None and record.trade_date < start:
            continue
        if end is not None and record.trade_date > end:
            continue
        for row in loaded:
            rows.append(_row_to_tradeability(row))
    rows.sort(key=lambda r: (r.stock_id, r.trade_date))
    fingerprint_payload = [
        (r.trade_date.isoformat(), r.row_count, r.sha256) for r, _ in pairs
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return rows, TradeabilitySource(
        data_root=source_root,
        manifest_fingerprint=fingerprint,
        date_count=len(pairs),
        row_count=len(rows),
        verified=True,
    )


__all__ = [
    "TradeabilityRow",
    "TradeabilitySource",
    "discover_tradeability_records",
    "load_tradeability",
]
