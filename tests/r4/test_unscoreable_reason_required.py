import pytest

from seatflow.r4.scoring import build_scoreable_daily_pool


def test_missing_market_data_requires_explicit_unscoreable_reason() -> None:
    with pytest.raises(ValueError):
        build_scoreable_daily_pool(
            trade_date="d1",
            persistent_candidates=["A"],
            previous_holdings=[],
            current_event_candidates=["A"],
            recent_event_candidates=[],
            market_features={},
        )

