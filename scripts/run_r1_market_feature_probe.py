"""Evaluate audited market-normalized flow features on exact R1 rolling folds."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.run_r1_cross_sectional_rank_probe import (
    _daily_metrics,
    _file_sha256,
    _hash_json,
    _model,
    _positive_probabilities,
    _summaries,
)


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    feature_block = cache_manifest.get("added_feature_block")
    if feature_block not in {
        "daily_amount_normalized_institution_flow",
        "point_in_time_sw2021_industry",
        "point_in_time_market_state",
    }:
        raise ValueError("R1 market probe requires an audited point-in-time matrix")
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 market matrix hash mismatch")
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if (
        source_result.get("dataset_sha256")
        != cache_manifest.get("source_dataset_sha256")
        or dataset_manifest.get("dataset_sha256")
        != cache_manifest.get("source_dataset_sha256")
    ):
        raise ValueError("R1 market inputs do not share the frozen dataset")

    matrix = np.load(matrix_path)
    x = np.asarray(matrix["x"], dtype=np.float64)
    sample_ids = np.asarray(matrix["sample_ids"])
    as_of_dates = np.asarray(matrix["as_of_dates"])
    labels = np.asarray(matrix["labels"], dtype=np.int8)
    returns = np.asarray(matrix["excess_returns"], dtype=np.float64)
    folds = _build_folds(
        sample_ids,
        as_of_dates,
        [str(value) for value in dataset_manifest["source_as_of_dates"]],
        min_train_dates=args.min_train_dates,
        n_folds=3,
        max_declared_horizon=5,
    )
    if args.min_train_dates == int(source_result["min_train_dates"]):
        _verify_frozen_fold_membership(folds, source_result["folds"], sample_ids)

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name in ("logistic_regression", "mlp", "lightgbm"):
        for seed in args.seeds:
            for fold in folds:
                train_indexes = fold["train_indexes"]
                test_indexes = fold["test_indexes"]
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
                            "added_feature_block": cache_manifest[
                                "added_feature_block"
                            ],
                            "fitted_sample_ids_sha256": _string_hash(
                                sample_ids[train_indexes]
                            ),
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
                        "transform": {
                            "mean": scaler.mean_.tolist(),
                            "scale": scaler.scale_.tolist(),
                            "fitted_sample_count": len(train_indexes),
                            "fitted_sample_ids_sha256": _string_hash(
                                sample_ids[train_indexes]
                            ),
                        },
                        "metrics": metrics,
                    }
                )

    summaries = _summaries(rows)
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "experiment": f"R1_POINT_IN_TIME_FEATURES_MIN{args.min_train_dates}D",
        "claim_tested": (
            f"the audited {feature_block} feature block improves the "
            "strong tabular baselines"
        ),
        "single_changed_variable": (
            "feature_set"
            if args.min_train_dates == int(source_result["min_train_dates"])
            else "minimum_training_history"
        ),
        "added_feature_block": cache_manifest["added_feature_block"],
        "added_feature_names": cache_manifest["added_feature_names"],
        "min_train_dates": args.min_train_dates,
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
                    "train_sample_count",
                    "test_sample_count",
                    "train_sample_ids_sha256",
                    "test_sample_ids_sha256",
                )
            }
            for fold in folds
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


def _build_folds(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    source_calendar: list[str],
    *,
    min_train_dates: int,
    n_folds: int,
    max_declared_horizon: int,
) -> list[dict[str, object]]:
    unique_dates = sorted({str(value) for value in as_of_dates})
    if len(unique_dates) < min_train_dates + n_folds:
        raise ValueError("not enough R1 dates for requested folds")
    test_chunks = _contiguous_chunks(unique_dates[min_train_dates:], n_folds)
    calendar_index = {day: index for index, day in enumerate(source_calendar)}
    label_end_dates: list[str] = []
    for value in as_of_dates:
        index = calendar_index[str(value)] + 2
        if index >= len(source_calendar):
            raise ValueError("R1 H1 label end falls outside the source calendar")
        label_end_dates.append(source_calendar[index])
    folds: list[dict[str, object]] = []
    date_values = np.asarray([str(value) for value in as_of_dates])
    label_end_values = np.asarray(label_end_dates)
    for index, test_dates in enumerate(test_chunks):
        test_start, test_end = test_dates[0], test_dates[-1]
        cutoff = (
            date.fromisoformat(test_start) - timedelta(days=max_declared_horizon)
        ).isoformat()
        train_indexes = np.flatnonzero(
            (date_values < cutoff) & (label_end_values < cutoff)
        )
        test_indexes = np.flatnonzero(
            (date_values >= test_start) & (date_values <= test_end)
        )
        if not len(train_indexes) or not len(test_indexes):
            raise ValueError(f"R1 fold {index} is empty")
        folds.append(
            {
                "fold_id": f"fold_{index}",
                "train_start": min(str(value) for value in date_values[train_indexes]),
                "train_end": max(str(value) for value in date_values[train_indexes]),
                "test_start": test_start,
                "test_end": test_end,
                "train_sample_count": len(train_indexes),
                "test_sample_count": len(test_indexes),
                "train_sample_ids_sha256": _string_hash(
                    sample_ids[train_indexes]
                ),
                "test_sample_ids_sha256": _string_hash(sample_ids[test_indexes]),
                "train_indexes": train_indexes,
                "test_indexes": test_indexes,
            }
        )
    return folds


def _contiguous_chunks(values: list[str], count: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    start = 0
    for index in range(count):
        remaining_values = len(values) - start
        remaining_chunks = count - index
        size = (remaining_values + remaining_chunks - 1) // remaining_chunks
        chunk = values[start : start + size]
        if not chunk:
            raise ValueError("R1 rolling fold date chunk is empty")
        chunks.append(chunk)
        start += size
    return chunks


def _verify_frozen_fold_membership(
    generated: list[dict[str, object]],
    frozen: object,
    sample_ids: np.ndarray,
) -> None:
    if not isinstance(frozen, list) or len(generated) != len(frozen):
        raise ValueError("frozen R1 folds are invalid")
    for generated_fold, frozen_fold in zip(generated, frozen, strict=True):
        if not isinstance(frozen_fold, dict):
            raise ValueError("frozen R1 fold is invalid")
        train_indexes = generated_fold["train_indexes"]
        test_indexes = generated_fold["test_indexes"]
        if not isinstance(train_indexes, np.ndarray) or not isinstance(
            test_indexes, np.ndarray
        ):
            raise TypeError("generated R1 fold indexes are invalid")
        if set(str(value) for value in sample_ids[train_indexes]) != set(
            str(value) for value in frozen_fold["train_sample_ids"]
        ):
            raise ValueError("generated R1 train fold differs from frozen membership")
        if set(str(value) for value in sample_ids[test_indexes]) != set(
            str(value) for value in frozen_fold["test_sample_ids"]
        ):
            raise ValueError("generated R1 test fold differs from frozen membership")


def _string_hash(values: np.ndarray) -> str:
    return sha256(
        json.dumps([str(value) for value in values], separators=(",", ":")).encode()
    ).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--r1-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-train-dates", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030]
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
