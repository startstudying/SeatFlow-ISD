"""Evaluate corrected raw-amount aggregation against the frozen R1 folds."""

from __future__ import annotations

import argparse
import json
import pickle
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
    _summaries,
)


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        cache_manifest.get("event_amount_semantics")
        != "inverse_signed_log_before_aggregation"
    ):
        raise ValueError("R1 amount probe requires corrected amount semantics")
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 amount matrix hash mismatch")
    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if (
        source_result.get("dataset_sha256")
        != cache_manifest.get("source_dataset_sha256")
    ):
        raise ValueError("R1 amount matrix does not match the frozen baseline result")

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
    for model_name in ("logistic_regression", "mlp", "lightgbm"):
        for seed in args.seeds:
            for fold in source_result["folds"]:
                train_indexes = _indexes(fold["train_sample_ids"], row_by_id)
                test_indexes = _indexes(fold["test_sample_ids"], row_by_id)
                scaler = StandardScaler().fit(x[train_indexes])
                train_x = scaler.transform(x[train_indexes])
                test_x = scaler.transform(x[test_indexes])
                model = _model(model_name, seed)
                model.fit(train_x, labels[train_indexes])
                probabilities = _positive_probabilities(model, test_x)
                metrics = _daily_metrics(
                    as_of_dates[test_indexes],
                    returns[test_indexes],
                    probabilities,
                    args.top_k,
                )
                checkpoint_path = (
                    output / f"{model_name}_{fold['fold_id']}_seed{seed}.pkl"
                )
                checkpoint_path.write_bytes(
                    pickle.dumps(
                        {
                            "scaler": scaler,
                            "model": model,
                            "feature_names": cache_manifest["feature_names"],
                            "event_amount_semantics": cache_manifest[
                                "event_amount_semantics"
                            ],
                        },
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold["fold_id"],
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": _file_sha256(checkpoint_path),
                        "metrics": metrics,
                    }
                )

    summaries = _summaries(rows)
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "experiment": "R1_RAW_AMOUNT_AGGREGATION_PROBE",
        "claim_tested": (
            "restoring raw edge amounts before aggregation repairs the semantic "
            "meaning of the frozen R1 amount features"
        ),
        "single_changed_variable": "event_amount_aggregation_semantics",
        "event_amount_semantics": cache_manifest["event_amount_semantics"],
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
        "summaries": summaries,
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


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
