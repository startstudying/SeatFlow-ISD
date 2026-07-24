"""Run the fixed equal-weight R1 ensemble across five seeds and three folds."""

from __future__ import annotations

import argparse
import json
import pickle
from math import isfinite
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.run_r1_cross_sectional_rank_probe import (
    _daily_metrics,
    _file_sha256,
    _hash_json,
    _indexes,
    _model,
    _positive_probabilities,
)

ENSEMBLE_MEMBERS: tuple[str, ...] = (
    "logistic_regression",
    "mlp",
    "lightgbm",
)


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest.get("added_feature_block") != "point_in_time_daily_basic":
        raise ValueError("R1 ensemble requires the audited daily_basic matrix")
    if cache_manifest.get("daily_basic_sample_coverage") != 1.0:
        raise ValueError("R1 ensemble requires complete daily_basic sample coverage")
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 ensemble matrix hash mismatch")
    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if (
        source_result.get("dataset_sha256")
        != cache_manifest.get("source_dataset_sha256")
    ):
        raise ValueError("R1 ensemble matrix does not match the frozen baseline result")

    matrix = np.load(matrix_path)
    x = np.asarray(matrix["x"], dtype=np.float64)
    sample_ids = [str(value) for value in matrix["sample_ids"]]
    as_of_dates = np.asarray(matrix["as_of_dates"])
    labels = np.asarray(matrix["labels"], dtype=np.int8)
    returns = np.asarray(matrix["excess_returns"], dtype=np.float64)
    row_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for fold in source_result["folds"]:
            train_indexes = _indexes(fold["train_sample_ids"], row_by_id)
            test_indexes = _indexes(fold["test_sample_ids"], row_by_id)
            scaler = StandardScaler().fit(x[train_indexes])
            train_x = scaler.transform(x[train_indexes])
            test_x = scaler.transform(x[test_indexes])
            models: dict[str, object] = {}
            member_probabilities: list[np.ndarray] = []
            for model_name in ENSEMBLE_MEMBERS:
                model = _model(model_name, seed)
                model.fit(train_x, labels[train_indexes])
                models[model_name] = model
                member_probabilities.append(
                    _positive_probabilities(model, test_x)
                )
            probabilities = np.mean(member_probabilities, axis=0)
            metrics = _daily_metrics(
                as_of_dates[test_indexes],
                returns[test_indexes],
                probabilities,
                args.top_k,
            )
            checkpoint_path = output / f"equal_weight_{fold['fold_id']}_seed{seed}.pkl"
            checkpoint_path.write_bytes(
                pickle.dumps(
                    {
                        "scaler": scaler,
                        "models": models,
                        "members": ENSEMBLE_MEMBERS,
                        "weights": [1 / len(ENSEMBLE_MEMBERS)] * len(ENSEMBLE_MEMBERS),
                        "feature_names": cache_manifest["feature_names"],
                        "fitted_sample_ids_sha256": _hash_json(
                            [sample_ids[index] for index in train_indexes]
                        ),
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
            rows.append(
                {
                    "model": "equal_weight_tabular_ensemble",
                    "seed": seed,
                    "fold": fold["fold_id"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                    "metrics": metrics,
                }
            )

    summary = _gate_summary(rows)
    performance_gate_passed = bool(summary["performance_gate_passed"])
    blocked_reasons = []
    if not performance_gate_passed:
        blocked_reasons.append("r1_reproducible_positive_signal_not_established")
    if source_result.get("catboost_disposition") != "completed":
        blocked_reasons.append("required_catboost_baseline_not_completed")
    r1_gate_passed = not blocked_reasons
    payload = {
        "status": (
            "evaluated_r1_passed"
            if r1_gate_passed
            else "evaluated_fail_closed"
        ),
        "model_valid": False,
        "r1_performance_gate_passed": performance_gate_passed,
        "r1_gate_passed": r1_gate_passed,
        "blocked_reasons": blocked_reasons,
        "downstream_blocked_reasons": [
            "missing_industry_coverage",
            "missing_concept_coverage",
            "missing_sealed_future_blind_evaluation",
        ],
        "experiment": "R1_EQUAL_WEIGHT_TABULAR_ENSEMBLE_856D_5SEED",
        "claim_tested": (
            "a fixed equal-weight aggregation of the required available "
            "tabular families is reproducibly positive"
        ),
        "single_changed_variable": "model_aggregation",
        "ensemble_members": list(ENSEMBLE_MEMBERS),
        "ensemble_weights": [1 / len(ENSEMBLE_MEMBERS)] * len(ENSEMBLE_MEMBERS),
        "source_r1_result": str(args.r1_result),
        "source_r1_result_sha256": source_result["sha256"],
        "matrix_manifest_sha256": cache_manifest["sha256"],
        "dataset_sha256": cache_manifest["source_dataset_sha256"],
        "seeds": args.seeds,
        "top_k": args.top_k,
        "folds": [
            {
                key: fold[key]
                for key in (
                    "fold_id",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                )
            }
            for fold in source_result["folds"]
        ],
        "rows": rows,
        "summary": summary,
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "r1_performance_gate_passed": performance_gate_passed,
                "r1_gate_passed": r1_gate_passed,
                "blocked_reasons": blocked_reasons,
                "summary": summary,
                "sha256": payload["sha256"],
            },
            indent=2,
        )
    )


def _gate_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    top_k = [
        float(row["metrics"]["daily_top_k_mean_excess_return"])  # type: ignore[index]
        for row in rows
    ]
    rank_ic = [
        float(row["metrics"]["daily_rank_ic_mean"])  # type: ignore[index]
        for row in rows
    ]
    seeds = {int(row["seed"]) for row in rows}
    folds = {str(row["fold"]) for row in rows}
    positive_run_fraction = sum(value > 0 for value in top_k) / len(top_k)
    top_k_mean = sum(top_k) / len(top_k)
    rank_ic_mean = sum(rank_ic) / len(rank_ic)
    passed = (
        len(seeds) >= 5
        and len(folds) >= 3
        and all(isfinite(value) for value in top_k + rank_ic)
        and top_k_mean > 0
        and positive_run_fraction >= 2 / 3
        and rank_ic_mean > 0
    )
    return {
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "run_count": len(rows),
        "daily_top_k_mean_excess_return": top_k_mean,
        "daily_top_k_min_excess_return": min(top_k),
        "daily_top_k_max_excess_return": max(top_k),
        "positive_run_fraction": positive_run_fraction,
        "daily_rank_ic_mean": rank_ic_mean,
        "performance_gate_passed": passed,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--r1-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030]
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
