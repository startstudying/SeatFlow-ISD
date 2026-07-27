from seatflow.r4.persistent_universe import build_persistent_candidate_universe


def test_duplicate_candidates_are_deduped() -> None:
    days = build_persistent_candidate_universe(
        daily_events={"d1": ["A", "A"], "d2": ["A"]},
        previous_holdings={"d2": ["A", "A"]},
        trading_calendar=["d1", "d2"],
    )

    assert [candidate.stock_id for candidate in days[1].candidates] == ["A"]
    assert days[1].candidates[0].reason == "multiple_reasons"

