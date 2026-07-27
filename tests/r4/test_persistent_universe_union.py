from seatflow.r4.persistent_universe import build_persistent_candidate_universe


def test_persistent_universe_unions_current_recent_and_holdings() -> None:
    days = build_persistent_candidate_universe(
        daily_events={
            "d1": ["A", "B"],
            "d2": ["C"],
            "d3": [],
        },
        previous_holdings={
            "d1": [],
            "d2": ["A", "H"],
            "d3": ["H"],
        },
        trading_calendar=["d1", "d2", "d3"],
    )

    day2 = {candidate.stock_id: candidate for candidate in days[1].candidates}
    assert set(day2) == {"A", "B", "C", "H"}
    assert day2["C"].reason == "current_event"
    assert day2["B"].reason == "recent_event"
    assert day2["H"].reason == "existing_holding"
    assert day2["A"].reason == "multiple_reasons"


def test_persistent_universe_includes_watchlist_candidates() -> None:
    days = build_persistent_candidate_universe(
        daily_events={
            "d1": ["A", "B"],
            "d2": ["C"],
            "d3": [],
            "d4": [],
        },
        previous_holdings={
            "d1": [],
            "d2": [],
            "d3": [],
            "d4": [],
        },
        watchlist={
            "d1": [],
            "d2": ["W1", "W2"],
            "d3": ["W1"],
            "d4": [],
        },
        trading_calendar=["d1", "d2", "d3", "d4"],
    )

    day2 = {candidate.stock_id: candidate for candidate in days[1].candidates}
    assert set(day2) == {"A", "B", "C", "W1", "W2"}
    assert day2["C"].reason == "current_event"
    assert day2["W1"].reason == "watchlist"
    assert day2["W2"].reason == "watchlist"
    assert day2["A"].reason == "recent_event"
    assert day2["B"].reason == "recent_event"

    # d3 has no events, but W1 is on watchlist
    day3 = {candidate.stock_id: candidate for candidate in days[2].candidates}
    assert set(day3) == {"W1", "A", "B", "C"}  # W1 via watchlist; A/B/C within 5-day age window
    assert day3["W1"].reason == "watchlist"
    assert day3["W1"].is_watchlist_candidate is True

