from __future__ import annotations

from math import log1p

import numpy as np

from scripts.materialize_r1_market_state_features import _market_state_by_date


def test_market_state_uses_only_same_and_prior_dates() -> None:
    rows = {
        "2026-01-01": [
            {"close_price": "110", "pre_close": "100", "amount": "2"},
            {"close_price": "90", "pre_close": "100", "amount": "3"},
        ],
        "2026-01-02": [
            {"close_price": "120", "pre_close": "100", "amount": "4"},
            {"close_price": "110", "pre_close": "100", "amount": "1"},
        ],
    }

    state = _market_state_by_date(rows)

    np.testing.assert_allclose(
        state["2026-01-01"],
        (0.0, 0.0, 0.5, 0.1, log1p(5000.0), 0.0, 0.0, 0.0),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        state["2026-01-02"],
        (
            0.15,
            0.15,
            1.0,
            0.05,
            log1p(5000.0),
            0.15,
            0.15,
            0.075,
        ),
        atol=1e-15,
    )
