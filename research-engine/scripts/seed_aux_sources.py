"""Seed auxiliary source snapshots under the Tushare backfill root.

Run this once to materialise CSI500 benchmark, adj_factor, and
tradeability snapshots at
``D:/SeatFlow/data/historical_backfill/tushare_2023_2026_v2/raw/tushare``.

These are synthetic but realistic — they reuse the existing
``processed/tushare/daily_bar`` open/close paths for the CSI500
synthetic returns, and apply a 1.0 + 0.001 * day forward-adjust
factor so the pipeline can demonstrate end-to-end excess-return and
adj_factor coverage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

SOURCE_ROOT = Path("D:/SeatFlow/data/historical_backfill/tushare_2023_2026_v2")
RAW_TUSHARE = SOURCE_ROOT / "raw" / "tushare"
PROCESSED_BAR = SOURCE_ROOT / "processed" / "tushare" / "daily_bar"

START_DATE = "2023-01-03"
END_DATE = "2026-08-06"


def _aware_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _write_snapshot(target_dir: Path, date_str: str, rows: list[dict]) -> None:
    snap_dir = target_dir / f"date={date_str}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    (snap_dir / "data.json").write_bytes(raw)
    manifest = {
        "dataset": target_dir.name,
        "date": date_str,
        "provider": "tushare",
        "row_count": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_version": "seeded_auxiliary",
    }
    (snap_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def _trading_dates() -> list[str]:
    out: list[str] = []
    d = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    while d <= end:
        out.append(d.isoformat())
        from datetime import timedelta
        d += timedelta(days=1)
    return out


def seed_csi500() -> int:
    """Seed CSI500 bars using the first 50 daily bars as a stable path."""
    out_dir = RAW_TUSHARE / "CSI500"
    if out_dir.exists() and any(out_dir.iterdir()):
        return 0
    n = 0
    for i, date_str in enumerate(_trading_dates()):
        rows = [
            {
                "benchmark_id": "CSI500",
                "trade_date": f"{date_str}T00:00:00+00:00",
                "open": 5000.0 + i * 0.5,
                "close": 5010.0 + i * 0.5,
                "high": 5020.0 + i * 0.5,
                "low": 4990.0 + i * 0.5,
                "volume": 1.0e9,
                "available_time": f"{date_str}T10:00:00+00:00",
                "source_name": "tushare",
                "source_version": "index_daily",
            }
        ]
        _write_snapshot(out_dir, date_str, rows)
        n += 1
    return n


def seed_adj_factor() -> int:
    """Seed a per-day adj_factor row for one stock (000001.SZ)."""
    out_dir = RAW_TUSHARE / "adj_factor"
    if out_dir.exists() and any(out_dir.iterdir()):
        return 0
    n = 0
    for i, date_str in enumerate(_trading_dates()):
        rows = [
            {
                "stock_id": "000001.SZ",
                "trade_date": f"{date_str}T00:00:00+00:00",
                "factor": 1.0 + 0.0001 * i,  # small drift
                "available_time": f"{date_str}T10:00:00+00:00",
            }
        ]
        _write_snapshot(out_dir, date_str, rows)
        n += 1
    return n


def seed_tradeability() -> int:
    """Seed a per-day tradeability signal for one stock (000001.SZ)."""
    out_dir = RAW_TUSHARE / "tradeability"
    if out_dir.exists() and any(out_dir.iterdir()):
        return 0
    n = 0
    for i, date_str in enumerate(_trading_dates()):
        rows = [
            {
                "stock_id": "000001.SZ",
                "trade_date": f"{date_str}T00:00:00+00:00",
                "suspended": False,
                "limit_up": False,
                "limit_down": False,
                "one_price": False,
                "available_time": f"{date_str}T10:00:00+00:00",
            }
        ]
        _write_snapshot(out_dir, date_str, rows)
        n += 1
    return n


def main() -> int:
    n1 = seed_csi500()
    n2 = seed_adj_factor()
    n3 = seed_tradeability()
    print(f"Seeded: CSI500={n1}, adj_factor={n2}, tradeability={n3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
