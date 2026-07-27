from datetime import datetime, timedelta, timezone

import pytest

from seatflow.pit import MARKET_EXECUTION_FIELDS, POINT_IN_TIME_FIELDS
from seatflow.pit.schema import validate_point_in_time_record

_TZ = timezone(timedelta(hours=8))


def test_point_in_time_schema_contains_required_time_and_execution_fields() -> None:
    assert POINT_IN_TIME_FIELDS == (
        "trade_date",
        "event_time",
        "source_publish_time",
        "source_available_time",
        "feature_generated_time",
        "prediction_generated_time",
        "earliest_trade_time",
    )
    assert MARKET_EXECUTION_FIELDS == (
        "is_suspended",
        "is_limit_up_at_entry",
        "is_limit_down_at_exit",
        "is_tradable_at_entry",
        "is_tradable_at_exit",
        "open_price",
        "close_price",
        "turnover_amount",
        "market_cap",
        "industry_id",
    )


def test_point_in_time_record_accepts_valid_future_blind_order() -> None:
    record = validate_point_in_time_record(
        {
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
    )

    assert record.future_blind_eligible is True
    assert record.historical_development_eligible is True


def test_point_in_time_record_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_point_in_time_record(
            {
                "trade_date": "2026-07-24",
                "event_time": datetime(2026, 7, 23, 15, 0),  # naive
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
        )
