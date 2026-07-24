from __future__ import annotations

import numpy as np

from scripts.materialize_r1_history_features import _historical_stock_features


def test_history_features_respect_stock_identity_and_trading_date_embargo() -> None:
    sample_ids = np.asarray(
        [
            "2026-01-01::000001.SZ",
            "2026-01-02::000002.SZ",
            "2026-01-03::000001.SZ",
            "2026-01-04::000001.SZ",
        ]
    )
    dates = np.asarray(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    )
    returns = np.asarray([0.10, 0.90, -0.20, 0.30])
    calendar = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]

    features = _historical_stock_features(
        sample_ids,
        dates,
        returns,
        calendar,
        embargo_trading_dates=2,
    )

    assert features[0, 0] == 0
    assert features[1, 0] == 0
    assert features[2, 0] == 1
    assert features[2, 3] == 0.10
    assert features[2, 6] == 1.0
    assert features[3, 0] == 1
    assert features[3, 3] == 0.10
