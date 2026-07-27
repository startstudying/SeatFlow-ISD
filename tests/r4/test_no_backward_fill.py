from seatflow.r4.decayed_seat_state import SeatEvent, propagate_decayed_seat_state


def test_future_event_is_not_backward_filled() -> None:
    states = propagate_decayed_seat_state(
        events_by_date={
            "d2": [
                SeatEvent(
                    trade_date="d2",
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

    assert states[0].last_seat_event_date is None
    assert states[0].decayed_seat_signal == 0.0

