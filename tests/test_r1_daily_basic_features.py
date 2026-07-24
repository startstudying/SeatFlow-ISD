from __future__ import annotations

from scripts.materialize_r1_daily_basic_features import _daily_basic_feature_row


def test_daily_basic_feature_row_adds_ratios_and_missing_flags() -> None:
    row = _daily_basic_feature_row(
        {
            "turnover_rate": 2.0,
            "turnover_rate_f": 4.0,
            "volume_ratio": None,
            "total_share": 100.0,
            "float_share": 80.0,
            "free_share": 40.0,
            "total_mv": 1000.0,
            "circ_mv": 600.0,
        }
    )

    assert row[:12] == (
        1.0,
        2.0,
        4.0,
        0.0,
        100.0,
        80.0,
        40.0,
        1000.0,
        600.0,
        0.6,
        0.8,
        0.5,
    )
    assert row[12:] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
