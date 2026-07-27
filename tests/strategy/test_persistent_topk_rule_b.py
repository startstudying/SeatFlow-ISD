from seatflow.strategy.persistent_topk import ScoredStock, apply_persistent_topk_rule_b


def test_rule_b_executes_cost_covered_replacement() -> None:
    decision = apply_persistent_topk_rule_b(
        trade_date="d1",
        previous_holdings=["A", "B", "C", "D", "E"],
        candidates=[
            ScoredStock("A", 0.01),
            ScoredStock("B", 0.02),
            ScoredStock("C", 0.03),
            ScoredStock("D", 0.04),
            ScoredStock("E", 0.05, sell_cost=0.001),
            ScoredStock("F", 0.10, buy_cost=0.001, slippage=0.001),
        ],
    )

    assert decision.executed_replacements == (("A", "F"),)
    assert decision.estimated_switch_cost == 0.002

