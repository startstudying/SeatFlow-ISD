from seatflow.evaluation.r3_6_replay import _turnover


def test_turnover_splits_buys_and_sells_without_double_counting() -> None:
    previous = {"A": 0.5, "B": 0.5}
    current = {"A": 0.5, "C": 0.5}

    buy, sell = _turnover(current, previous)

    assert buy == 0.5
    assert sell == 0.5
