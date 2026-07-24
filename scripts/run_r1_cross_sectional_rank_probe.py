"""Probe date-local cross-sectional feature ranks for the frozen R1 baselines."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from seatflow.research import _spearman


def main() -> None:
    args = _arguments()
    cache_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    matrix_path = Path(str(cache_manifest["matrix"]))
    if _file_sha256(matrix_path) != cache_manifest.get("matrix_sha256"):
        raise ValueError("R1 matrix cache hash mismatch")
    source_result = json.loads(args.r1_result.read_text(encoding="utf-8"))
    if (
        source_result.get("dataset_sha256")
        != cache_manifest.get("source_dataset_sha256")
    ):
        raise ValueError("R1 probe dataset does not match the frozen baseline result")

    matrix = np.load(matrix_path)
    raw_x = np.asarray(matrix["x"], dtype=np.float64)
    sample_ids = [str(value) for value in matrix["sample_ids"]]
    as_of_dates = np.asarray(matrix["as_of_dates"])
    labels = np.asarray(matrix["labels"], dtype=np.int8)
    returns = np.asarray(matrix["excess_returns"], dtype=np.float64)
    x = _cross_sectional_ranks(raw_x, as_of_dates)
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
                            "feature_representation": (
                                "date_local_cross_sectional_percentile_rank"
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
                        "metrics": metrics,
                    }
                )

    summaries = _summaries(rows)
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "experiment": "R1_CROSS_SECTIONAL_RANK_PROBE",
        "claim_tested": (
            "date-local feature ranks reduce regime-scale drift while preserving "
            "the frozen daily ranking task"
        ),
        "single_changed_variable": "feature_representation",
        "feature_representation": "date_local_cross_sectional_percentile_rank",
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


def _cross_sectional_ranks(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or dates.shape != (values.shape[0],):
        raise ValueError("R1 cross-sectional ranks require aligned 2D values and dates")
    transformed = np.empty_like(values, dtype=np.float64)
    for day in np.unique(dates):
        indexes = np.flatnonzero(dates == day)
        denominator = max(len(indexes) - 1, 1)
        for column in range(values.shape[1]):
            column_values = values[indexes, column]
            order = np.argsort(column_values, kind="stable")
            ranks = np.empty(len(indexes), dtype=np.float64)
            start = 0
            while start < len(order):
                end = start + 1
                while (
                    end < len(order)
                    and column_values[order[end]] == column_values[order[start]]
                ):
                    end += 1
                ranks[order[start:end]] = (start + end - 1) / 2.0
                start = end
            transformed[indexes, column] = (
                ranks / denominator if len(indexes) > 1 else 0.5
            )
    return transformed


def _model(name: str, seed: int) -> object:
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=200, random_state=seed, solver="saga"
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            random_state=seed,
            verbosity=-1,
        )
    raise ValueError(f"unknown R1 classification model: {name}")


def _positive_probabilities(model: object, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)  # type: ignore[attr-defined]
    classes = [int(value) for value in model.classes_]  # type: ignore[attr-defined]
    if 1 not in classes:
        raise ValueError("R1 classifier did not fit a positive class")
    return np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)


def _indexes(values: object, row_by_id: dict[str, int]) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("R1 fold sample ids must be a list")
    try:
        return np.asarray([row_by_id[str(value)] for value in values], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"R1 matrix cache is missing sample {error.args[0]}") from error


def _daily_metrics(
    dates: np.ndarray,
    returns: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> dict[str, float | int]:
    by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for day, value, score in zip(dates, returns, scores, strict=True):
        by_date[str(day)].append((float(score), float(value)))
    daily_returns: list[float] = []
    daily_ic: list[float] = []
    equity, peak, maximum_drawdown = 1.0, 1.0, 0.0
    for day in sorted(by_date):
        values = sorted(by_date[day], key=lambda item: item[0], reverse=True)
        selected = values[:top_k]
        selected_return = sum(value for _, value in selected) / len(selected)
        daily_returns.append(selected_return)
        ic = _spearman(
            [score for score, _ in values], [value for _, value in values]
        )
        if ic is not None:
            daily_ic.append(ic)
        equity *= 1.0 + selected_return
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    return {
        "sample_count": len(scores),
        "daily_rank_ic_mean": sum(daily_ic) / len(daily_ic),
        "daily_top_k_mean_excess_return": sum(daily_returns) / len(daily_returns),
        "daily_top_k_max_drawdown": maximum_drawdown,
        "positive_day_fraction": sum(value > 0 for value in daily_returns)
        / len(daily_returns),
    }


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summaries: list[dict[str, object]] = []
    for model, model_rows in grouped.items():
        top_k = [
            float(row["metrics"]["daily_top_k_mean_excess_return"])  # type: ignore[index]
            for row in model_rows
        ]
        rank_ic = [
            float(row["metrics"]["daily_rank_ic_mean"])  # type: ignore[index]
            for row in model_rows
        ]
        summaries.append(
            {
                "model": model,
                "run_count": len(model_rows),
                "daily_top_k_mean_excess_return": sum(top_k) / len(top_k),
                "daily_top_k_min_excess_return": min(top_k),
                "daily_top_k_max_excess_return": max(top_k),
                "positive_run_fraction": sum(value > 0 for value in top_k)
                / len(top_k),
                "daily_rank_ic_mean": sum(rank_ic) / len(rank_ic),
            }
        )
    return summaries


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
