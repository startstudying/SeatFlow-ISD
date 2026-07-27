from datetime import datetime, timedelta, timezone

import pytest

from seatflow.pit.schema import validate_point_in_time_record

_TZ = timezone(timedelta(hours=8))


def _base_row() -> dict[str, object]:
    return {
        "trade_date": "2026-07-24",
        "event_time": datetime(2026, 7, 23, 15, 0, tzinfo=_TZ),
        "source_publish_time": datetime(2026, 7, 23, 17, 0, tzinfo=_TZ),
        "source_available_time": datetime(2026, 7, 23, 17, 1, tzinfo=_TZ),
        "feature_generated_time": datetime(2026, 7, 24, 8, 0, tzinfo=_TZ),
        "prediction_generated_time": datetime(2026, 7, 24, 8, 30, tzinfo=_TZ),
        "earliest_trade_time": datetime(2026, 7, 24, 9, 30, tzinfo=_TZ),
        "is_suspended": False,
        "is_limit_up_at_entry": False,
        "is_limit_down_at_exit": False,
        "is_tradable_at_entry": True,
        "is_tradable_at_exit": True,
        "open_price": 10.0,
        "close_price": 10.5,
        "turnover_amount": 1000000.0,
        "market_cap": 500000000.0,
        "industry_id": "801010",
    }


def test_source_available_must_not_be_after_prediction() -> None:
    row = _base_row()
    row["source_available_time"] = datetime(2026, 7, 24, 8, 31, tzinfo=_TZ)

    with pytest.raises(ValueError, match="source_available_time"):
        validate_point_in_time_record(row)


def test_prediction_must_be_before_earliest_trade() -> None:
    row = _base_row()
    row["prediction_generated_time"] = datetime(2026, 7, 24, 9, 30, tzinfo=_TZ)

    with pytest.raises(ValueError, match="prediction_generated_time"):
        validate_point_in_time_record(row)
