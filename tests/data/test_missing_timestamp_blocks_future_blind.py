from datetime import datetime, timedelta, timezone

from seatflow.pit.schema import validate_point_in_time_record

_TZ = timezone(timedelta(hours=8))


def test_missing_precise_timestamp_blocks_future_blind_without_fabrication() -> None:
    record = validate_point_in_time_record(
        {
            "trade_date": "2026-07-24",
            "event_time": None,
            "source_publish_time": None,
            "source_available_time": None,
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

    assert record.source_available_time is None
    assert record.future_blind_eligible is False
    assert record.historical_development_eligible is True
