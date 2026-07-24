from __future__ import annotations

from scripts.materialize_r1_market_normalized_features import (
    _normalized_market_row,
)


def test_market_row_normalizes_raw_flow_by_audited_daily_amount() -> None:
    row = _normalized_market_row(
        200.0,
        50.0,
        (100.0, 200.0),
        40.0,
        10.0,
        30.0,
    )

    assert row == (200.0, 50.0, 4.0 / 3.0, 0.2, 0.05, 0.15, 0.25, 0.6)
