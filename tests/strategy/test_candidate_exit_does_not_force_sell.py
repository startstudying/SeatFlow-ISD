from seatflow.strategy.persistent_topk import ScoredStock, apply_persistent_topk_rule_b


def test_candidate_exit_reason_is_ignored_by_forced_exit_logic() -> None:
    decision = apply_persistent_topk_rule_b(
        trade_date="d1",
        previous_holdings=["A", "B", "C", "D", "E"],
        candidates=[ScoredStock(stock, 0.01) for stock in ["A", "B", "C", "D", "E"]],
        forced_exit_reasons={"A": "candidate_exit"},
    )

    assert decision.forced_exits == ()

