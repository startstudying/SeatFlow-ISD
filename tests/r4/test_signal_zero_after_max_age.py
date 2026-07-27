from seatflow.r4.decayed_seat_state import SeatEvent, propagate_decayed_seat_state


def test_signal_zero_after_maximum_event_age() -> None:
    calendar = [f"d{i}" for i in range(7)]
    states = propagate_decayed_seat_state(
        events_by_date={
            "d0": [
                SeatEvent(
                    trade_date="d0",
                    stock_id="A",
                    net_buy_amount=1.0,
                    seat_count=1,
                    institution_ratio=0.0,
                    seat_concentration=1.0,
                    seat_signal=4.0,
                )
            ]
        },
        universe_by_date={day: ["A"] for day in calendar},
        trading_calendar=calendar,
    )

    assert states[-1].days_since_last_seat_event == 6
    assert states[-1].decayed_seat_signal == 0.0

