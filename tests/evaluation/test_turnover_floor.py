from seatflow.evaluation.turnover_floor import (
    analyze_candidate_turnover_floor,
)


def test_candidate_exit_turnover_floor_uses_full_candidate_pool() -> None:
    daily, _summary = analyze_candidate_turnover_floor(
        daily_holdings={
            "2026-01-01": ["A", "B", "C", "D", "E"],
            "2026-01-02": ["A", "F", "G", "H", "I"],
        },
        daily_candidates={
            "2026-01-01": ["A", "B", "C", "D", "E", "X"],
            "2026-01-02": ["A", "F", "G", "H", "I"],
        },
        portfolio_size=5,
    )

    assert daily[0].candidate_exit_count == 4
    assert daily[0].candidate_exit_turnover_floor == 0.8


def test_turnover_floor_fail_closed_when_candidate_pool_missing() -> None:
    try:
        analyze_candidate_turnover_floor(
            daily_holdings={"2026-01-01": ["A"]},
            daily_candidates={},
        )
    except ValueError as exc:
        assert "missing_candidates" in str(exc)
    else:
        raise AssertionError("missing candidate pool must fail closed")


def test_turnover_floor_summary_reports_max_possible_reduction() -> None:
    _daily, summary = analyze_candidate_turnover_floor(
        daily_holdings={
            "2026-01-01": ["A", "B", "C", "D", "E"],
            "2026-01-02": ["F", "G", "H", "I", "J"],
        },
        daily_candidates={
            "2026-01-01": ["A", "B", "C", "D", "E"],
            "2026-01-02": ["A", "B", "F", "G", "H", "I", "J"],
        },
        portfolio_size=5,
    )

    assert summary["mean_candidate_exit_turnover_floor"] == 0.6
    assert summary["maximum_possible_reduction_under_rule_A"] == 0.4
