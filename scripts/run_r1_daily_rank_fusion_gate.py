"""Evaluate fixed daily-rank fusion on the audited R1 tabular matrix."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.run_r1_cross_sectional_rank_probe import (
    _cross_sectional_ranks,
    _daily_metrics,
    _file_sha256,
    _hash_json,
    _indexes,
    _model,
    _positive_probabilities,
)
from scripts.run_r1_equal_weight_ensemble_gate import (
    ENSEMBLE_MEMBERS,
    _gate_summary,
)


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest.get("added_feature_block") != "point_in_time_daily_basic":
        raise ValueError("R1 daily-rank fusion requires the audited daily_basic matrix")
    if cache_manifest.get("daily_basic_sample_coverage") != 1.0:
        raise ValueError("R1 daily-rank fusion requires complete daily_basic coverage")
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 daily-rank fusion matrix hash mismatch")

    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if source_result.get("dataset_sha256") != cache_manifest.get(
        "source_dataset_sha256"
    ):
        raise ValueError("R1 daily-rank fusion does not share the frozen dataset")

    matrix = np.load(matrix_path)
    x = np.asarray(matrix["x"], dtype=np.float64)
    sample_ids = [str(value) for value in matrix["sample_ids"]]
    as_of_dates = np.asarray(matrix["as_of_dates"])
    labels = np.asarray(matrix["labels"], dtype=np.int8)
    returns = np.asarray(matrix["excess_returns"], dtype=np.float64)
    row_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    args.output_root.mkdir(parents=True, exist_ok=True)
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
            scores = _daily_rank_fusion(
                np.column_stack(member_probabilities),
                as_of_dates[test_indexes],
            )
            metrics = _daily_metrics(
                as_of_dates[test_indexes],
                returns[test_indexes],
                scores,
                args.top_k,
            )
            checkpoint_path = (
                args.output_root
                / f"daily_rank_fusion_{fold['fold_id']}_seed{seed}.pkl"
            )
            checkpoint_path.write_bytes(
                pickle.dumps(
                    {
                        "scaler": scaler,
                        "models": models,
                        "members": ENSEMBLE_MEMBERS,
                        "aggregation": "equal_mean_of_date_local_percentile_ranks",
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
                    "model": "daily_rank_tabular_fusion",
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
    payload = {
        "status": (
            "evaluated_r1_passed"
            if not blocked_reasons
            else "evaluated_fail_closed"
        ),
        "model_valid": False,
        "r1_performance_gate_passed": performance_gate_passed,
        "r1_gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "experiment": "R1_E5C_DAILY_RANK_FUSION",
        "claim_tested": (
            "fixed date-local percentile-rank fusion stabilizes the required "
            "available tabular direction baselines"
        ),
        "single_changed_variable": "model_aggregation",
        "aggregation": "equal_mean_of_date_local_percentile_ranks",
        "ensemble_members": list(ENSEMBLE_MEMBERS),
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
    (args.output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "sha256": payload["sha256"]}, indent=2))


def _daily_rank_fusion(
    member_scores: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    if member_scores.ndim != 2 or member_scores.shape[1] < 2:
        raise ValueError("R1 daily-rank fusion requires at least two score columns")
    if dates.shape != (member_scores.shape[0],):
        raise ValueError("R1 daily-rank fusion requires aligned dates")
    ranked = _cross_sectional_ranks(member_scores, dates)
    return np.mean(ranked, axis=1)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--r1-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    return parser.parse_args()


if __name__ == "__main__":
    main()
