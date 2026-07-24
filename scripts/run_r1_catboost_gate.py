"""Run the required CatBoost R1 baseline on the audited point-in-time matrix."""

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
    _positive_probabilities,
)
from scripts.run_r1_equal_weight_ensemble_gate import _gate_summary
from scripts.run_r1_market_feature_probe import _build_folds
from scripts.run_r1_v2_tabular_baselines import _model


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    feature_block = cache_manifest.get("added_feature_block")
    if feature_block not in {
        "point_in_time_daily_basic",
        "point_in_time_sw2021_industry",
        "point_in_time_market_state",
    }:
        raise ValueError("R1 CatBoost requires an audited point-in-time matrix")
    if (
        feature_block == "point_in_time_daily_basic"
        and cache_manifest.get("daily_basic_sample_coverage") != 1.0
    ):
        raise ValueError("R1 CatBoost requires complete daily_basic sample coverage")
    if (
        feature_block in {"point_in_time_sw2021_industry", "point_in_time_market_state"}
        and float(cache_manifest.get("industry_sample_coverage", 0.0)) < 0.99
    ):
        raise ValueError("R1 CatBoost requires at least 99% industry coverage")
    if (
        feature_block == "point_in_time_market_state"
        and cache_manifest.get("market_state_date_coverage") != 1.0
    ):
        raise ValueError("R1 CatBoost requires complete market-state coverage")
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 CatBoost matrix hash mismatch")

    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if source_result.get("dataset_sha256") != cache_manifest.get(
        "source_dataset_sha256"
    ):
        raise ValueError("R1 CatBoost inputs do not share the frozen dataset")

    matrix = np.load(matrix_path)
    x = np.asarray(matrix["x"], dtype=np.float64)
    feature_names = [str(value) for value in cache_manifest["feature_names"]]
    sample_ids = [str(value) for value in matrix["sample_ids"]]
    as_of_dates = np.asarray(matrix["as_of_dates"])
    labels = np.asarray(matrix["labels"], dtype=np.int8)
    returns = np.asarray(matrix["excess_returns"], dtype=np.float64)
    row_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    source_min_train_dates = int(source_result["min_train_dates"])
    if args.min_train_dates == source_min_train_dates:
        folds = source_result["folds"]
    else:
        if args.dataset_manifest is None:
            raise ValueError(
                "R1 CatBoost requires the dataset manifest for regenerated folds"
            )
        dataset_manifest = json.loads(
            args.dataset_manifest.read_text(encoding="utf-8")
        )
        if dataset_manifest.get("dataset_sha256") != cache_manifest.get(
            "source_dataset_sha256"
        ):
            raise ValueError("R1 CatBoost dataset manifest hash does not match")
        folds = _build_folds(
            np.asarray(sample_ids),
            as_of_dates,
            [str(value) for value in dataset_manifest["source_as_of_dates"]],
            min_train_dates=args.min_train_dates,
            n_folds=3,
            max_declared_horizon=5,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for fold in folds:
            if "train_indexes" in fold:
                train_indexes = np.asarray(fold["train_indexes"], dtype=np.int64)
                test_indexes = np.asarray(fold["test_indexes"], dtype=np.int64)
            else:
                train_indexes = _indexes(fold["train_sample_ids"], row_by_id)
                test_indexes = _indexes(fold["test_sample_ids"], row_by_id)
            train_x, test_x, scaler, cat_features, fitted_feature_names = (
                _prepare_features(
                    x,
                    feature_names,
                    train_indexes,
                    test_indexes,
                    native_industry_category=args.native_industry_category,
                )
            )
            model = _catboost_model(seed, deterministic=args.deterministic)
            model.fit(  # type: ignore[attr-defined]
                train_x,
                labels[train_indexes],
                cat_features=cat_features,
            )
            scores = _positive_probabilities(model, test_x)
            metrics = _daily_metrics(
                as_of_dates[test_indexes],
                returns[test_indexes],
                scores,
                args.top_k,
            )
            checkpoint_path = (
                args.output_root / f"catboost_{fold['fold_id']}_seed{seed}.pkl"
            )
            checkpoint_path.write_bytes(
                pickle.dumps(
                    {
                        "scaler": scaler,
                        "model": model,
                        "feature_names": fitted_feature_names,
                        "cat_features": cat_features,
                        "fitted_sample_ids_sha256": _hash_json(
                            [sample_ids[index] for index in train_indexes]
                        ),
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            )
            rows.append(
                {
                    "model": "catboost",
                    "seed": seed,
                    "fold": fold["fold_id"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                    "metrics": metrics,
                }
            )

    summary = _gate_summary(rows)
    performance_gate_passed = bool(summary["performance_gate_passed"])
    payload = {
        "status": (
            "evaluated_r1_performance_passed"
            if performance_gate_passed
            else "evaluated_fail_closed"
        ),
        "model_valid": False,
        "r1_performance_gate_passed": performance_gate_passed,
        "r1_gate_passed": performance_gate_passed,
        "catboost_disposition": "completed",
        "blocked_reasons": (
            []
            if performance_gate_passed
            else ["r1_reproducible_positive_signal_not_established"]
        ),
        "experiment": "R1_CATBOOST_DAILY_BASIC_856D_5SEED",
        "claim_tested": (
            "the required CatBoost family establishes a reproducibly positive "
            "signal on the complete point-in-time R1 matrix"
        ),
        "single_changed_variable": (
            "industry_encoding"
            if args.native_industry_category
            else (
                "training_randomness"
                if args.deterministic
                else (
                    "model_family"
                    if args.min_train_dates == source_min_train_dates
                    else "minimum_training_history"
                )
            )
        ),
        "deterministic_training": args.deterministic,
        "native_industry_category": args.native_industry_category,
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
                )
            }
            for fold in folds
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--r1-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-train-dates", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--native-industry-category", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
    )
    return parser.parse_args()


def _prepare_features(
    x: np.ndarray,
    feature_names: list[str],
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    *,
    native_industry_category: bool,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[int], list[str]]:
    industry_indexes = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("industry::sw2021_l1::")
    ]
    if not native_industry_category:
        scaler = StandardScaler().fit(x[train_indexes])
        return (
            scaler.transform(x[train_indexes]),
            scaler.transform(x[test_indexes]),
            scaler,
            [],
            feature_names,
        )
    if not industry_indexes:
        raise ValueError("native CatBoost industry encoding requires L1 one-hot features")
    continuous_indexes = [
        index for index in range(x.shape[1]) if index not in industry_indexes
    ]
    scaler = StandardScaler().fit(x[train_indexes][:, continuous_indexes])
    train_numeric = scaler.transform(x[train_indexes][:, continuous_indexes])
    test_numeric = scaler.transform(x[test_indexes][:, continuous_indexes])
    category_names = [
        feature_names[index].removeprefix("industry::sw2021_l1::")
        for index in industry_indexes
    ]

    def with_category(values: np.ndarray, indexes: np.ndarray) -> np.ndarray:
        output = np.empty((len(indexes), values.shape[1] + 1), dtype=object)
        output[:, :-1] = values
        one_hot = x[indexes][:, industry_indexes]
        output[:, -1] = [
            category_names[int(np.argmax(row))] if float(np.max(row)) > 0 else "MISSING"
            for row in one_hot
        ]
        return output

    fitted_names = [feature_names[index] for index in continuous_indexes] + [
        "industry::sw2021_l1"
    ]
    category_index = len(fitted_names) - 1
    return (
        with_category(train_numeric, train_indexes),
        with_category(test_numeric, test_indexes),
        scaler,
        [category_index],
        fitted_names,
    )


def _catboost_model(seed: int, *, deterministic: bool) -> object:
    if not deterministic:
        return _model("catboost", seed)
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=seed,
        bootstrap_type="No",
        random_strength=0.0,
        rsm=1.0,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )


if __name__ == "__main__":
    main()
