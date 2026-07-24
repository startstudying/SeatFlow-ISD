from __future__ import annotations

import numpy as np

from scripts.run_r1_market_feature_probe import _build_folds


def test_matrix_folds_apply_h1_label_end_and_max_horizon_purge() -> None:
    calendar = [f"2026-01-{day:02d}" for day in range(1, 13)]
    dates = np.asarray(calendar[:10])
    sample_ids = np.asarray([f"{day}::000001.SZ" for day in dates])

    folds = _build_folds(
        sample_ids,
        dates,
        calendar,
        min_train_dates=4,
        n_folds=3,
        max_declared_horizon=1,
    )

    assert folds[0]["test_start"] == "2026-01-05"
    assert folds[0]["train_end"] == "2026-01-01"
    assert folds[0]["train_sample_count"] == 1
