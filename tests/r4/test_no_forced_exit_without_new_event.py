from seatflow.r4.scoring import build_scoreable_daily_pool


def test_holding_without_new_event_keeps_a_scoreable_row() -> None:
    rows = build_scoreable_daily_pool(
        trade_date="d2",
        persistent_candidates=["B"],
        previous_holdings=["A"],
        current_event_candidates=["B"],
        recent_event_candidates=[],
        market_features={"A": {"close_price": 10.0}, "B": {"close_price": 11.0}},
    )

    holding = next(row for row in rows if row.stock_id == "A")
    assert holding.scoreable is True
    assert holding.unscoreable_reason is None
