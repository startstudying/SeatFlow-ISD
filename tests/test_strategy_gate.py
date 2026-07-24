from seatflow.evaluation.strategy_gate import (
    evaluate_strategy_gate,
    select_unique_strategy,
)


def _runs(
    net: list[float],
    *,
    gross: float,
    turnover: float,
    cost: float,
) -> list[dict[str, float | int]]:
    return [
        {
            "fold": index // 3,
            "seed": 2026 + index % 3,
            "mean_net_spread": value,
            "mean_gross_spread": gross,
            "mean_turnover": turnover,
            "mean_cost_diff": cost,
        }
        for index, value in enumerate(net)
    ]


def test_gate_requires_all_frozen_economic_checks() -> None:
    baseline = _runs(
        [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001],
        gross=0.001,
        turnover=1.0,
        cost=0.001,
    )
    strategy = _runs(
        [0.001, 0.001, -0.0001, 0.001, 0.001, -0.0001],
        gross=0.0005,
        turnover=0.6,
        cost=0.0006,
    )

    result = evaluate_strategy_gate(strategy, baseline)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_selection_is_fail_closed_when_no_strategy_passes() -> None:
    selection = select_unique_strategy({"S0/P1": {"passed": False, "metrics": {}}})

    assert selection["selected_strategy"] is None
    assert selection["task48_allowed"] is False
    assert selection["status"] == "negative_result_under_current_data"
