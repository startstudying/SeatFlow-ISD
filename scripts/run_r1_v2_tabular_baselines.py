"""Run V2-only table baselines with fold-local transforms and fail-closed evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from collections import defaultdict
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from seatflow.datasets.pyg import EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA, TensorizedWindowSample
from seatflow.evaluation.dt_baselines import _feature_vector
from seatflow.evaluation.dt_rolling import build_dt_rolling_folds
from seatflow.evaluation.v2_metrics import V2EvaluationRow, evaluate_v2_rows

_FEATURE_CACHE: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {}
BASELINE_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "lightgbm",
    "catboost",
    "mlp",
)


def main() -> None:
    args = _arguments()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("label_schema_version") != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA:
        raise ValueError("R1 V2 baselines reject a non-V2 dataset manifest")
    samples = pickle.loads((args.dataset_root / "v2-tensor-dataset.pkl").read_bytes())
    if any(
        sample.label_schema_version != EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA for sample in samples
    ):
        raise ValueError("R1 V2 baselines reject mixed-label tensor samples")
    folds = build_dt_rolling_folds(
        samples, selection_horizon=1, n_folds=3, min_train_dates=args.min_train_dates
    )
    available = {
        "logistic_regression": True,
        "mlp": True,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "catboost": importlib.util.find_spec("catboost") is not None,
    }
    rows: list[dict[str, object]] = []
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_MODELS:
        if not available[name]:
            continue
        for seed in args.seeds:
            for fold in folds:
                trained, metric_rows, transform = _fit_predict(name, fold.train, fold.test, seed)
                model_path = out / f"{name}_{fold.fold_id}_seed{seed}.pkl"
                model_path.write_bytes(pickle.dumps(trained, protocol=pickle.HIGHEST_PROTOCOL))
                metrics = evaluate_v2_rows(metric_rows, top_k=5)
                metrics.update(_daily_strategy_metrics(metric_rows, 5))
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "fold": fold.fold_id,
                        "checkpoint": str(model_path),
                        "checkpoint_sha256": _hash_file(model_path),
                        "transform": transform,
                        "metrics": metrics,
                    }
                )
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "experiment": "R1_V2_TABULAR_BASELINES",
        "single_changed_variable": "model_family",
        "dataset_manifest_sha256": manifest["sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "label_schema_version": EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        "min_train_dates": args.min_train_dates,
        "seeds": args.seeds,
        "dependency_availability": available,
        "catboost_disposition": _catboost_disposition(
            available["catboost"],
            rows,
            seed_count=len(args.seeds),
            fold_count=len(folds),
        ),
        "folds": [fold.to_dict() for fold in folds],
        "rows": rows,
        "code_sha256": _hash_file(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "catboost": payload["catboost_disposition"]}))


def _fit_predict(
    name: str,
    train: tuple[TensorizedWindowSample, ...],
    test: tuple[TensorizedWindowSample, ...],
    seed: int,
) -> tuple[object, tuple[V2EvaluationRow, ...], dict[str, object]]:
    train_x, feature_names = _matrix(train)
    test_x, test_names = _matrix(test)
    if feature_names != test_names:
        raise ValueError("R1 feature schema differs between train and test")
    scaler = StandardScaler().fit(train_x)
    x_train, x_test = scaler.transform(train_x), scaler.transform(test_x)
    y_train = np.asarray([sample.labels[1] for sample in train])
    model = _model(name, seed)
    model.fit(x_train, y_train)
    probabilities = _three_class_probabilities(model, x_test)
    metric_rows = tuple(
        V2EvaluationRow(
            sample.as_of_date.isoformat(), sample.labels[1], sample.returns[1], probability
        )
        for sample, probability in zip(test, probabilities, strict=True)
    )
    return (
        {"scaler": scaler, "model": model, "feature_names": feature_names},
        metric_rows,
        {
            "feature_names": feature_names,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "fitted_sample_count": len(train),
            "fitted_sample_ids_sha256": _hash_json([sample.sample_id for sample in train]),
        },
    )


def _model(name: str, seed: int) -> object:
    if name == "logistic_regression":
        return LogisticRegression(max_iter=200, random_state=seed, solver="saga")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
        )
    raise ValueError(f"unknown R1 baseline model: {name}")


def _catboost_disposition(
    available: bool,
    rows: list[dict[str, object]],
    *,
    seed_count: int,
    fold_count: int,
) -> str:
    if not available:
        return "dependency_unavailable"
    completed = sum(row.get("model") == "catboost" for row in rows)
    return "completed" if completed == seed_count * fold_count else "incomplete"


def _matrix(samples: tuple[TensorizedWindowSample, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    names: tuple[str, ...] | None = None
    rows: list[tuple[float, ...]] = []
    for sample in samples:
        cached = _FEATURE_CACHE.get(sample.sample_id)
        if cached is None:
            stock_names = sample.graph.metadata.get("feature_schema", {}).get("stock", [])
            vector = _feature_vector(sample) + tuple(
                float(value)
                for value in sample.graph.data["stock"]
                .x[sample.target_node_index]
                .detach()
                .cpu()
                .tolist()
            )
            current = tuple(
                f"event_profile::{index}" for index in range(len(vector) - len(stock_names))
            ) + tuple(f"stock::{name}" for name in stock_names)
            _FEATURE_CACHE[sample.sample_id] = vector, current
        else:
            vector, current = cached
        if names is None:
            names = current
        elif names != current:
            raise ValueError("R1 tensor feature schema is not stable")
        if any(not isfinite(value) for value in vector):
            raise ValueError(f"non-finite R1 feature: {sample.sample_id}")
        rows.append(vector)
    return np.asarray(rows, dtype=float), names or ()


def _three_class_probabilities(
    model: object, values: np.ndarray
) -> tuple[tuple[float, float, float], ...]:
    probabilities = model.predict_proba(values)  # type: ignore[attr-defined]
    classes = [int(value) for value in model.classes_]  # type: ignore[attr-defined]
    return tuple(
        tuple(float(row[classes.index(label)]) if label in classes else 0.0 for label in (-1, 0, 1))
        for row in probabilities
    )


def _daily_strategy_metrics(rows: tuple[V2EvaluationRow, ...], top_k: int) -> dict[str, float]:
    by_date: dict[str, list[V2EvaluationRow]] = defaultdict(list)
    for row in rows:
        by_date[row.as_of_date].append(row)
    daily = [
        sum(
            item.excess_return
            for item in sorted(group, key=lambda item: item.probabilities[2], reverse=True)[:top_k]
        )
        / min(top_k, len(group))
        for group in by_date.values()
    ]
    equity, peak, maximum_drawdown = 1.0, 1.0, 0.0
    for value in daily:
        equity *= 1 + value
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1)
    return {
        "daily_top_k_mean_excess_return": sum(daily) / len(daily),
        "daily_top_k_max_drawdown": maximum_drawdown,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-train-dates", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hash_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
