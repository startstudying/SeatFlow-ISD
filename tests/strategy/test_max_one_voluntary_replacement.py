from seatflow.strategy.persistent_topk import ScoredStock, apply_persistent_topk_rule_b


def test_rule_b_executes_at_most_one_voluntary_replacement() -> None:
    decision = apply_persistent_topk_rule_b(
        trade_date="d1",
        previous_holdings=["A", "B", "C", "D", "E"],
        candidates=[
            ScoredStock("A", 0.0),
            ScoredStock("B", 0.0),
            ScoredStock("C", 0.0),
            ScoredStock("D", 0.0),
            ScoredStock("E", 0.0),
            ScoredStock("F", 1.0),
            ScoredStock("G", 0.9),
        ],
    )

    assert len(decision.executed_replacements) == 1

