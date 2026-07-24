from __future__ import annotations

from datetime import date

from scripts.backfill_r1_daily_basic import _canonical_rows


def test_daily_basic_canonicalization_preserves_missingness() -> None:
    rows, missingness = _canonical_rows(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260105",
                "turnover_rate": 2.5,
                "turnover_rate_f": 3.0,
                "volume_ratio": None,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 60.0,
                "total_mv": 1000.0,
                "circ_mv": 800.0,
            }
        ],
        date(2026, 1, 5),
    )

    assert rows[0]["stock_id"] == "000001.SZ"
    assert rows[0]["volume_ratio"] is None
    assert missingness["volume_ratio"] == 1
    assert missingness["turnover_rate"] == 0
