from __future__ import annotations

from scripts.materialize_r1_summary_flow_features import _summary_feature_row


def test_summary_feature_row_separates_total_and_institution_flow() -> None:
    row = _summary_feature_row(
        row_count=2.0,
        summary_buy=80.0,
        summary_sell=20.0,
        summary_net=60.0,
        daily_amount=200.0,
        institution_buy=40.0,
        institution_sell=10.0,
    )

    assert row == (
        1.0,
        2.0,
        80.0,
        20.0,
        60.0,
        100.0,
        0.4,
        0.1,
        0.3,
        0.6,
        0.5,
        0.5,
        0.5,
    )
