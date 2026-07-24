from __future__ import annotations

import numpy as np

from scripts.run_r1_cross_sectional_rank_probe import _cross_sectional_ranks


def test_cross_sectional_ranks_are_date_local_and_average_ties() -> None:
    values = np.asarray(
        [
            [10.0, 4.0],
            [20.0, 4.0],
            [30.0, 9.0],
            [100.0, 2.0],
            [50.0, 3.0],
        ]
    )
    dates = np.asarray(["2026-01-01"] * 3 + ["2026-01-02"] * 2)

    ranked = _cross_sectional_ranks(values, dates)

    np.testing.assert_allclose(
        ranked,
        np.asarray(
            [
                [0.0, 0.25],
                [0.5, 0.25],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )
