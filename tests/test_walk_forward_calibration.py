from seatflow.evaluation.walk_forward_calibration import (
    ridge_walk_forward_expected_returns,
)


def test_walk_forward_excludes_unmatured_and_future_labels() -> None:
    rows = [
        {
            "prediction_date": "2026-01-01",
            "label_end_date": "2026-01-03",
            "stock_id": "A",
            "score": 1.0,
            "realized_return": 10.0,
        },
        {
            "prediction_date": "2026-01-02",
            "label_end_date": "2026-01-02",
            "stock_id": "B",
            "score": 2.0,
            "realized_return": 20.0,
        },
        {
            "prediction_date": "2026-01-03",
            "label_end_date": "2026-01-04",
            "stock_id": "C",
            "score": 3.0,
            "realized_return": 30.0,
        },
        {
            "prediction_date": "2026-01-04",
            "label_end_date": "2026-01-05",
            "stock_id": "D",
            "score": 4.0,
            "realized_return": 40.0,
        },
    ]

    calibrated = ridge_walk_forward_expected_returns(rows, alpha=1.0)

    assert [row.matured_training_rows for row in calibrated] == [0, 0, 1, 2]
    assert calibrated[0].expected_gross_return == 0.0
    assert calibrated[1].expected_gross_return == 0.0
    assert calibrated[2].expected_gross_return == 20.0
    assert calibrated[3].expected_gross_return != 30.0
