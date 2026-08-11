"""CSI500 (or any index) benchmark loader for SeatFlow-ISD V1.

This module is the formal path to eliminate the
``FORMAL_BENCHMARK_PRESENT`` blocker: it loads a verified benchmark
source and produces the same shape of bars as
:mod:`seatflow.dataset.isd_adapter.ISDMarketBar`, which the existing
``compute_market_excess_path`` consumes directly.

The expected on-disk layout follows the same ``date=YYYY-MM-DD`` +
``data.json`` + ``manifest.json`` convention as the rest of the
Tushare backfill. The benchmark id is the directory name
(e.g. ``CSI500``). The rows must carry at least ``benchmark_id``,
``trade_date``, ``open``, ``close``, ``high``, ``low``, ``volume``,
``available_time`` so that ``ISDMarketBar`` is produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from seatflow.dataset.isd_adapter import ISDMarketBar
from seatflow.dataset.source_snapshots import (
    SnapshotRecord,
    discover_and_load_records,
)
from seatflow.domain.time import assert_aware


@dataclass(frozen=True)
class BenchmarkBar:
    """A single daily benchmark bar."""

    benchmark_id: str
    stock_id: str  # always == benchmark_id for convenience
    trade_date: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float
    available_time: datetime
    source_lineage: str
    source_version: str

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")
        if self.open <= 0 or self.close <= 0 or self.high <= 0 or self.low <= 0:
            raise ValueError(
                f"benchmark bar prices must be strictly positive for {self.benchmark_id}"
            )

    def to_isd_market_bar(self) -> ISDMarketBar:
        """Cast to ``ISDMarketBar`` for ``compute_market_excess_path``."""
        return ISDMarketBar(
            stock_id=self.benchmark_id,
            trade_date=self.trade_date,
            open=self.open,
            close=self.close,
            high=self.high,
            low=self.low,
            volume=self.volume,
            available_time=self.available_time,
            source_lineage=self.source_lineage,
            source_version=self.source_version,
        )


@dataclass(frozen=True)
class BenchmarkSource:
    """Verified benchmark metadata."""

    benchmark_id: str
    data_root: Path
    manifest_fingerprint: str
    date_count: int
    row_count: int
    verified: bool


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert_aware(parsed, "datetime")
    return parsed


def _row_to_bar(benchmark_id: str, row: Mapping[str, Any]) -> BenchmarkBar:
    return BenchmarkBar(
        benchmark_id=benchmark_id,
        stock_id=benchmark_id,
        trade_date=_parse_dt(row["trade_date"]),
        open=float(row.get("open", 0.0) or 0.0),
        close=float(row.get("close", 0.0) or 0.0),
        high=float(row.get("high", 0.0) or 0.0),
        low=float(row.get("low", 0.0) or 0.0),
        volume=float(row.get("volume", 0.0) or 0.0),
        available_time=_parse_dt(row["available_time"]),
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=str(row.get("source_version", "index_daily")),
    )


def discover_benchmark_records(
    benchmark_root: Path,
    benchmark_id: str,
) -> list[tuple[SnapshotRecord, list[dict[str, Any]]]]:
    """Discover and load every verified benchmark snapshot.

    The benchmark root must contain a ``date=YYYY-MM-DD`` directory
    hierarchy with ``data.json`` + ``manifest.json`` per date, the
    same convention as the rest of the Tushare backfill.
    """
    dataset_dir = benchmark_root / benchmark_id
    if not dataset_dir.exists():
        return []
    return discover_and_load_records(dataset_dir.parent, Path(dataset_dir.name))


def load_benchmark(
    benchmark_root: Path,
    benchmark_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[BenchmarkBar], BenchmarkSource]:
    """Load verified benchmark bars and return a sorted list + metadata.

    Raises ``FileNotFoundError`` if no snapshots exist for the
    ``benchmark_id`` and ``ValueError`` on any hash mismatch.
    """
    pairs = discover_benchmark_records(benchmark_root, benchmark_id)
    if not pairs:
        raise FileNotFoundError(
            f"no verified snapshots for benchmark {benchmark_id!r} at {benchmark_root}"
        )
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    bars: list[BenchmarkBar] = []
    for record, rows in pairs:
        if start is not None and record.trade_date < start:
            continue
        if end is not None and record.trade_date > end:
            continue
        for row in rows:
            bars.append(_row_to_bar(benchmark_id, row))
    bars.sort(key=lambda b: b.trade_date)
    fingerprint_payload = [
        (r.trade_date.isoformat(), r.row_count, r.sha256) for r, _ in pairs
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    source = BenchmarkSource(
        benchmark_id=benchmark_id,
        data_root=benchmark_root,
        manifest_fingerprint=fingerprint,
        date_count=len(pairs),
        row_count=len(bars),
        verified=True,
    )
    return bars, source


def to_isd_market_bars(bars: Sequence[BenchmarkBar]) -> list[ISDMarketBar]:
    """Convert benchmark bars to the same shape that
    :func:`compute_market_excess_path` accepts.
    """
    return [b.to_isd_market_bar() for b in bars]


__all__ = [
    "BenchmarkBar",
    "BenchmarkSource",
    "discover_benchmark_records",
    "load_benchmark",
    "to_isd_market_bars",
]
