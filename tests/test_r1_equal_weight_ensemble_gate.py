from __future__ import annotations

from scripts.run_r1_equal_weight_ensemble_gate import _gate_summary


def test_r1_performance_gate_requires_positive_mean_and_two_thirds_runs() -> None:
    rows = []
    for seed in range(5):
        for fold, value in enumerate((0.003, 0.002, -0.001)):
            rows.append(
                {
                    "seed": seed,
                    "fold": f"fold_{fold}",
                    "metrics": {
                        "daily_top_k_mean_excess_return": value,
                        "daily_rank_ic_mean": 0.04,
                    },
                }
            )

    summary = _gate_summary(rows)

    assert summary["performance_gate_passed"] is True
    assert summary["positive_run_fraction"] == 2 / 3
