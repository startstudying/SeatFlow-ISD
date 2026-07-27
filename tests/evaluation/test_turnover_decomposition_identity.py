import pytest

from seatflow.evaluation.turnover_floor import TurnoverDecomposition


def test_turnover_decomposition_identity_accepts_exact_split() -> None:
    TurnoverDecomposition(
        candidate_exit_turnover=0.6,
        voluntary_rank_turnover=0.2,
        untradable_turnover=0.1,
        corporate_action_turnover=0.1,
    ).validate(observed_total_turnover=1.0)


def test_turnover_decomposition_identity_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="decomposition"):
        TurnoverDecomposition(
            candidate_exit_turnover=0.6,
            voluntary_rank_turnover=0.2,
            untradable_turnover=0.0,
            corporate_action_turnover=0.0,
        ).validate(observed_total_turnover=1.0)
