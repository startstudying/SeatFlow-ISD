from seatflow.strategy.persistent_topk import ScoredStock, apply_persistent_topk_rule_b


def test_switch_requires_return_gain_to_cover_cost() -> None:
    decision = apply_persistent_topk_rule_b(
        trade_date="d1",
        previous_holdings=["A", "B", "C", "D", "E"],
        candidates=[
            ScoredStock("A", 0.01),
            ScoredStock("B", 0.02),
            ScoredStock("C", 0.03),
            ScoredStock("D", 0.04),
            ScoredStock("E", 0.05),
            ScoredStock("F", 0.011, buy_cost=0.01),
        ],
    )

    assert decision.executed_replacements == ()
    assert decision.holdings == ("A", "B", "C", "D", "E")

