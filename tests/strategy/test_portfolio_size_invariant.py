import pytest

from seatflow.strategy.persistent_topk import ScoredStock, apply_persistent_topk_rule_b


def test_portfolio_size_remains_five_when_enough_candidates_exist() -> None:
    decision = apply_persistent_topk_rule_b(
        trade_date="d1",
        previous_holdings=["A", "B", "C"],
        candidates=[ScoredStock(stock, float(index)) for index, stock in enumerate("ABCDEFG")],
    )

    assert len(decision.holdings) == 5


def test_rule_b_fails_closed_when_portfolio_cannot_remain_full() -> None:
    with pytest.raises(ValueError, match="portfolio_size=5"):
        apply_persistent_topk_rule_b(
            trade_date="d1",
            previous_holdings=["A"],
            candidates=[ScoredStock("A", 0.1)],
        )


def test_retained_holding_requires_a_current_score() -> None:
    with pytest.raises(ValueError, match="require current scores"):
        apply_persistent_topk_rule_b(
            trade_date="d1",
            previous_holdings=["A", "B", "C", "D", "E"],
            candidates=[ScoredStock(stock, 0.1) for stock in ["B", "C", "D", "E", "F"]],
        )
