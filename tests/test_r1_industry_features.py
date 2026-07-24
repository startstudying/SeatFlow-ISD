from __future__ import annotations

import numpy as np

from scripts.materialize_r1_industry_features import (
    _active_membership,
    _index_memberships_by_stock,
    _industry_features,
    _is_current_membership,
)


def test_industry_membership_assignment_is_point_in_time_and_latest_wins() -> None:
    rows = [
        {
            "stock_id": "000001.SZ",
            "l1_code": "801010",
            "l2_code": "801011",
            "l3_code": "801012",
            "in_date": "2026-01-01",
            "out_date": "2026-02-01",
            "is_new": True,
        },
        {
            "stock_id": "000001.SZ",
            "l1_code": "801020",
            "l2_code": "801021",
            "l3_code": "801022",
            "in_date": "2026-01-20",
            "out_date": None,
            "is_new": True,
        },
    ]
    memberships = _index_memberships_by_stock(rows)

    assert _active_membership(memberships["000001.SZ"], "2025-12-31") is None
    assert _active_membership(memberships["000001.SZ"], "2026-01-10")[2] == "801010"
    assert _active_membership(memberships["000001.SZ"], "2026-01-20")[2] == "801020"
    assert _active_membership(memberships["000001.SZ"], "2026-02-01")[2] == "801020"


def test_industry_features_are_deterministic_and_preserve_alignment() -> None:
    sample_ids = np.asarray(
        [
            "2026-01-03::000002.SZ",
            "2026-01-10::000001.SZ",
            "2026-01-20::000001.SZ",
            "2026-01-20::000003.SZ",
        ]
    )
    as_of_dates = np.asarray(
        ["2026-01-03", "2026-01-10", "2026-01-20", "2026-01-20"]
    )
    rows = [
        {
            "stock_id": "000001.SZ",
            "l1_code": "801020",
            "l2_code": "801021",
            "l3_code": "801022",
            "in_date": "2026-01-20",
            "out_date": "",
            "is_new": True,
        },
        {
            "stock_id": "000002.SZ",
            "l1_code": "801010",
            "l2_code": "801011",
            "l3_code": "801012",
            "in_date": "2026-01-01",
            "out_date": None,
            "is_new": False,
        },
        {
            "stock_id": "000001.SZ",
            "l1_code": "801010",
            "l2_code": "801011",
            "l3_code": "801012",
            "in_date": "2026-01-01",
            "out_date": "2026-02-01",
            "is_new": True,
        },
    ]

    features, feature_names, coverage = _industry_features(sample_ids, as_of_dates, rows)

    assert feature_names == (
        "industry::available",
        "industry::membership_age_days",
        "industry::sw2021_l1::801010",
        "industry::sw2021_l1::801020",
    )
    assert coverage == 0.75
    np.testing.assert_allclose(
        features,
        np.asarray(
            [
                [1.0, 2.0, 1.0, 0.0],
                [1.0, 9.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_membership_state_parses_tushare_y_n_values() -> None:
    assert _is_current_membership("Y") is True
    assert _is_current_membership("N") is False
    assert _is_current_membership(False) is False
