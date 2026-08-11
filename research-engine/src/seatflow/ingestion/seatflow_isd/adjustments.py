"""Adjustment-factor and company-actions source for SeatFlow-ISD V1.

Prevents misinterpreting dividends / splits / rights issues as
returns. The expected on-disk layout follows the same
``date=YYYY-MM-DD`` + ``data.json`` + ``manifest.json`` convention
as the rest of the Tushare backfill.

Two related but distinct sources are accepted:

- ``adj_factor`` — per-(stock, date) cumulative adjustment factor
  (前复权 base). Multiplying the raw open/close by this factor
  reproduces the forward-adjusted price.
- ``company_actions`` — per-(stock, date) discrete event (dividend,
  split, rights issue). The schema is intentionally minimal so
  that downstream code can decide how to fold it into the price
  adjustment.

For V1.0 the only requirement is to have a verified source path
that downstream code can consult to fail-closed if adjustment
information is missing. The formal-validity gate is satisfied
as soon as either ``adj_factor`` or ``company_actions`` is present
with verified hashes.
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
class AdjFactorRow:
    """Per-(stock, date) cumulative adjustment factor.

    ``factor`` is the multiplier to convert raw open/close into
    forward-adjusted prices. ``factor == 1.0`` means no adjustment.
    """

    stock_id: str
    trade_date: datetime
    factor: float
    available_time: datetime

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")
        if self.factor <= 0:
            raise ValueError(
                f"adj_factor must be > 0 for {self.stock_id} on {self.trade_date}"
            )


@dataclass(frozen=True)
class CompanyActionRow:
    """A single discrete corporate action (dividend / split / etc.)."""

    stock_id: str
    ex_date: datetime
    action_type: str  # "dividend" | "split" | "rights" | "spinoff" | "other"
    amount: float  # shares-per-share for split, CNY/share for dividend
    available_time: datetime

    def __post_init__(self) -> None:
        assert_aware(self.ex_date, "ex_date")
        assert_aware(self.available_time, "available_time")
        if self.action_type not in {"dividend", "split", "rights", "spinoff", "other"}:
            raise ValueError(f"unknown action_type={self.action_type!r}")


@dataclass(frozen=True)
class AdjustmentSource:
    """Verified metadata for an adjustment-factor or company-action source."""

    source_id: str  # "adj_factor" or "company_actions"
    data_root: Path
    manifest_fingerprint: str
    date_count: int
    row_count: int
    verified: bool


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert_aware(parsed, "datetime")
    return parsed


def _row_to_adj_factor(row: Mapping[str, Any]) -> AdjFactorRow:
    return AdjFactorRow(
        stock_id=str(row["stock_id"]),
        trade_date=_parse_dt(row["trade_date"]),
        factor=float(row.get("factor", 1.0) or 1.0),
        available_time=_parse_dt(row["available_time"]),
    )


def _row_to_company_action(row: Mapping[str, Any]) -> CompanyActionRow:
    return CompanyActionRow(
        stock_id=str(row["stock_id"]),
        ex_date=_parse_dt(row.get("ex_date", row["trade_date"])),
        action_type=str(row.get("action_type", "other")),
        amount=float(row.get("amount", 0.0) or 0.0),
        available_time=_parse_dt(row["available_time"]),
    )


def discover_adjustment_records(
    source_root: Path,
    source_id: str,
) -> list[tuple[SnapshotRecord, list[dict[str, Any]]]]:
    """Discover and load every verified adjustment snapshot."""
    dataset_dir = source_root / source_id
    if not dataset_dir.exists():
        return []
    return discover_and_load_records(dataset_dir.parent, Path(dataset_dir.name))


def _build_metadata(
    pairs: list[tuple[SnapshotRecord, list[dict[str, Any]]]],
) -> tuple[str, int, int]:
    fingerprint_payload = [
        (r.trade_date.isoformat(), r.row_count, r.sha256) for r, _ in pairs
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    row_count = sum(len(rows) for _, rows in pairs)
    return fingerprint, len(pairs), row_count


def load_adj_factor(
    source_root: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[AdjFactorRow], AdjustmentSource]:
    """Load verified adj_factor rows."""
    pairs = discover_adjustment_records(source_root, "adj_factor")
    if not pairs:
        raise FileNotFoundError(
            f"no verified snapshots for adj_factor at {source_root}"
        )
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    rows: list[AdjFactorRow] = []
    for record, loaded in pairs:
        if start is not None and record.trade_date < start:
            continue
        if end is not None and record.trade_date > end:
            continue
        for row in loaded:
            rows.append(_row_to_adj_factor(row))
    rows.sort(key=lambda r: (r.stock_id, r.trade_date))
    fingerprint, date_count, row_count = _build_metadata(pairs)
    return rows, AdjustmentSource(
        source_id="adj_factor",
        data_root=source_root,
        manifest_fingerprint=fingerprint,
        date_count=date_count,
        row_count=row_count,
        verified=True,
    )


def load_company_actions(
    source_root: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[CompanyActionRow], AdjustmentSource]:
    """Load verified company_actions rows."""
    pairs = discover_adjustment_records(source_root, "company_actions")
    if not pairs:
        raise FileNotFoundError(
            f"no verified snapshots for company_actions at {source_root}"
        )
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    rows: list[CompanyActionRow] = []
    for record, loaded in pairs:
        if start is not None and record.trade_date < start:
            continue
        if end is not None and record.trade_date > end:
            continue
        for row in loaded:
            rows.append(_row_to_company_action(row))
    rows.sort(key=lambda r: (r.stock_id, r.ex_date))
    fingerprint, date_count, row_count = _build_metadata(pairs)
    return rows, AdjustmentSource(
        source_id="company_actions",
        data_root=source_root,
        manifest_fingerprint=fingerprint,
        date_count=date_count,
        row_count=row_count,
        verified=True,
    )


def adjust_price(
    raw_price: float,
    factor: float,
) -> float:
    """Multiply a raw price by an adj_factor to get the adjusted price."""
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    return raw_price * factor


__all__ = [
    "AdjFactorRow",
    "AdjustmentSource",
    "CompanyActionRow",
    "adjust_price",
    "discover_adjustment_records",
    "load_adj_factor",
    "load_company_actions",
]
