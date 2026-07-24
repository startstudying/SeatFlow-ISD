from __future__ import annotations

import numpy as np

from scripts.run_r1_catboost_gate import _prepare_features


def test_prepare_features_collapses_industry_one_hot_to_native_category() -> None:
    x = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
        ]
    )
    names = [
        "stock::return",
        "industry::sw2021_l1::801010.SI",
        "industry::sw2021_l1::801020.SI",
    ]

    train, test, _, cat_features, fitted_names = _prepare_features(
        x,
        names,
        np.asarray([0, 1]),
        np.asarray([2]),
        native_industry_category=True,
    )

    assert cat_features == [1]
    assert fitted_names == ["stock::return", "industry::sw2021_l1"]
    assert train[:, -1].tolist() == ["801010.SI", "801020.SI"]
    assert test[:, -1].tolist() == ["MISSING"]
