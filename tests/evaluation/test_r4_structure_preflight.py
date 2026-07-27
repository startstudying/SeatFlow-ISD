from seatflow.evaluation.structure_preflight import (
    evaluate_structure_preflight,
    fail_closed_structure_gate,
    structure_gate_decision,
)
from seatflow.evaluation.turnover_floor import DailyTurnoverFloor


def test_structure_preflight_fails_closed_without_daily_evidence() -> None:
    decision = fail_closed_structure_gate()

    assert decision["passed"] is False
    assert decision["evidence_complete"] is False
    assert decision["task59_allowed"] is False


def test_structure_preflight_passes_only_at_thirty_percent_reduction() -> None:
    floor = DailyTurnoverFloor(
        run_id="r",
        prediction_date="d2",
        previous_prediction_date="d1",
        portfolio_size=5,
        previous_holding_count=5,
        current_candidate_count=10,
        candidate_exit_count=1,
        candidate_exit_turnover_floor=0.2,
        total_turnover=0.8,
        removable_turnover_above_floor=0.6,
        candidate_pool_temporal_jaccard=0.5,
        candidate_pool_retention_rate=0.5,
    )
    result = evaluate_structure_preflight(
        p0_mean_turnover=1.0,
        turnover_floor_rows=[floor],
        persistent_pool_sizes=[10],
        persistent_pool_jaccards=[0.5],
        simulated_rule_b_turnovers=[0.7],
        retained_holding_counts=[4],
    )

    decision = structure_gate_decision(result)
    assert decision["passed"] is True
    assert decision["task59_allowed"] is True
    assert decision["task60_allowed"] is False


def test_structure_preflight_rejects_mismatched_daily_evidence() -> None:
    floor = DailyTurnoverFloor(
        run_id="r",
        prediction_date="d2",
        previous_prediction_date="d1",
        portfolio_size=5,
        previous_holding_count=5,
        current_candidate_count=10,
        candidate_exit_count=1,
        candidate_exit_turnover_floor=0.2,
        total_turnover=0.8,
        removable_turnover_above_floor=0.6,
        candidate_pool_temporal_jaccard=0.5,
        candidate_pool_retention_rate=0.5,
    )
    result = evaluate_structure_preflight(
        p0_mean_turnover=1.0,
        turnover_floor_rows=[floor],
        persistent_pool_sizes=[10, 11],
        persistent_pool_jaccards=[0.5],
        simulated_rule_b_turnovers=[0.7],
        retained_holding_counts=[4],
    )

    assert result.status == "insufficient_evidence"
    assert structure_gate_decision(result)["passed"] is False
