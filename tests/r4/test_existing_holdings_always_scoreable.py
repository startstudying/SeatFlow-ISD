from seatflow.r4.persistent_universe import build_persistent_candidate_universe


def test_existing_holding_stays_in_scoreable_pool_after_event_expiry() -> None:
    calendar = [f"d{i}" for i in range(1, 8)]
    days = build_persistent_candidate_universe(
        daily_events={"d1": ["A"]},
        previous_holdings={"d7": ["A"]},
        trading_calendar=calendar,
    )

    day7 = days[-1]
    assert [candidate.stock_id for candidate in day7.candidates] == ["A"]
    assert day7.candidates[0].is_existing_holding
    assert day7.candidates[0].reason == "existing_holding"
