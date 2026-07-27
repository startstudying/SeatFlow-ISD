from seatflow.r4.scoring import build_scoreable_daily_pool


def test_holding_outside_current_event_pool_is_scored() -> None:
    rows = build_scoreable_daily_pool(
        trade_date="d2",
        persistent_candidates=["B"],
        previous_holdings=["A"],
        current_event_candidates=["B"],
        recent_event_candidates=[],
        market_features={"A": {"close_price": 10.0}, "B": {"close_price": 11.0}},
    )

    by_stock = {row.stock_id: row for row in rows}
    assert by_stock["A"].scoreable
    assert by_stock["A"].is_existing_holding
    assert not by_stock["A"].is_current_event_candidate

