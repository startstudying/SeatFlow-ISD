from __future__ import annotations

from scripts.run_r1_v2_tabular_baselines import _catboost_disposition


def test_catboost_disposition_requires_every_seed_fold_run() -> None:
    rows = [
        {"model": "catboost", "seed": seed, "fold": fold}
        for seed in (2026, 2027)
        for fold in ("fold_0", "fold_1", "fold_2")
    ]

    assert (
        _catboost_disposition(True, rows, seed_count=2, fold_count=3)
        == "completed"
    )
    assert (
        _catboost_disposition(True, rows[:-1], seed_count=2, fold_count=3)
        == "incomplete"
    )
    assert (
        _catboost_disposition(False, [], seed_count=2, fold_count=3)
        == "dependency_unavailable"
    )
