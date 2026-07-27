from seatflow.r4.decayed_seat_state import half_life_decay


def test_half_life_decay_halves_at_half_life() -> None:
    assert half_life_decay(0) == 1.0
    assert half_life_decay(2) == 0.5

