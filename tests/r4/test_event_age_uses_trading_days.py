from seatflow.r4.persistent_universe import build_persistent_candidate_universe


def test_event_age_uses_calendar_index_not_natural_days() -> None:
    days = build_persistent_candidate_universe(
        daily_events={"2026-01-02": ["A"]},
        previous_holdings={},
        trading_calendar=["2026-01-02", "2026-01-05"],
    )

    assert days[1].candidates[0].stock_id == "A"
    assert days[1].candidates[0].event_age_trading_days == 1

