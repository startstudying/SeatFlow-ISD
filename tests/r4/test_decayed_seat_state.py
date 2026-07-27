from seatflow.r4.decayed_seat_state import SeatEvent, propagate_decayed_seat_state


def test_new_event_resets_age_and_state_propagates_forward() -> None:
    states = propagate_decayed_seat_state(
        events_by_date={
            "d1": [
                SeatEvent(
                    trade_date="d1",
                    stock_id="A",
                    net_buy_amount=100.0,
                    seat_count=2,
                    institution_ratio=0.5,
                    seat_concentration=0.7,
                    seat_signal=8.0,
                )
            ]
        },
        universe_by_date={"d1": ["A"], "d2": ["A"]},
        trading_calendar=["d1", "d2"],
    )

    assert states[0].has_new_seat_event
    assert states[0].days_since_last_seat_event == 0
    assert states[1].last_net_buy_amount == 100.0
    assert not states[1].has_new_seat_event

