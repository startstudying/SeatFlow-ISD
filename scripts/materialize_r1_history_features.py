"""Add point-in-time stock event-history features to an audited R1 matrix."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from hashlib import sha256
from pathlib import Path

import numpy as np

HISTORY_FEATURE_NAMES: tuple[str, ...] = (
    "stock_history::event_count_all",
    "stock_history::event_count_20d",
    "stock_history::event_count_60d",
    "stock_history::mean_excess_return_all",
    "stock_history::mean_excess_return_20d",
    "stock_history::mean_excess_return_60d",
    "stock_history::positive_rate_all",
    "stock_history::positive_rate_20d",
    "stock_history::positive_rate_60d",
)


def main() -> None:
    args = _arguments()
    source_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    source_matrix_path = Path(str(source_manifest["matrix"]))
    if _file_sha256(source_matrix_path) != source_manifest.get("matrix_sha256"):
        raise ValueError("R1 source matrix hash mismatch")
    dataset_manifest = json.loads(
        Path(str(source_manifest["source_manifest"])).read_text(encoding="utf-8")
    )
    source_calendar = [str(value) for value in dataset_manifest["source_as_of_dates"]]

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    history = _historical_stock_features(
        sample_ids,
        as_of_dates,
        excess_returns,
        source_calendar,
        embargo_trading_dates=args.embargo_trading_dates,
    )
    augmented = np.concatenate((x, history), axis=1)
    feature_names = [str(value) for value in source["feature_names"]] + list(
        HISTORY_FEATURE_NAMES
    )

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "matrix.npz"
    np.savez_compressed(
        matrix_path,
        x=augmented,
        sample_ids=sample_ids,
        as_of_dates=as_of_dates,
        labels=labels,
        excess_returns=excess_returns,
        feature_names=np.asarray(feature_names),
    )
    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "label_schema_version": source_manifest["label_schema_version"],
        "source_manifest": str(args.matrix_root / "manifest.json"),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_dataset_sha256": source_manifest["source_dataset_sha256"],
        "sample_count": len(sample_ids),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "added_feature_block": "point_in_time_stock_event_history",
        "added_feature_names": list(HISTORY_FEATURE_NAMES),
        "embargo_trading_dates": args.embargo_trading_dates,
        "sample_ids_sha256": source_manifest["sample_ids_sha256"],
        "matrix": str(matrix_path),
        "matrix_sha256": _file_sha256(matrix_path),
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": payload["sample_count"],
                "features": payload["feature_count"],
                "matrix_sha256": payload["matrix_sha256"],
            }
        )
    )


def _historical_stock_features(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    excess_returns: np.ndarray,
    source_calendar: list[str],
    *,
    embargo_trading_dates: int,
) -> np.ndarray:
    if embargo_trading_dates < 1:
        raise ValueError("R1 history features require a positive trading-date embargo")
    if not (
        sample_ids.shape == as_of_dates.shape == excess_returns.shape
        and sample_ids.ndim == 1
    ):
        raise ValueError("R1 history feature inputs must be aligned vectors")
    calendar_index = {day: index for index, day in enumerate(source_calendar)}
    if len(calendar_index) != len(source_calendar):
        raise ValueError("R1 source calendar contains duplicate dates")
    try:
        row_calendar_indexes = np.asarray(
            [calendar_index[str(day)] for day in as_of_dates], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"R1 source calendar is missing date {error.args[0]}") from error

    output = np.zeros((len(sample_ids), len(HISTORY_FEATURE_NAMES)), dtype=np.float64)
    histories: dict[str, list[tuple[int, float]]] = {}
    ordered_rows = sorted(
        range(len(sample_ids)),
        key=lambda row: (row_calendar_indexes[row], str(sample_ids[row])),
    )
    for row in ordered_rows:
        stock_id = str(sample_ids[row]).rsplit("::", maxsplit=1)[-1]
        current_index = int(row_calendar_indexes[row])
        history = histories.setdefault(stock_id, [])
        cutoff = current_index - embargo_trading_dates
        eligible_end = bisect_right(history, (cutoff, float("inf")))
        eligible = history[:eligible_end]
        recent_20 = [item for item in eligible if item[0] >= current_index - 20]
        recent_60 = [item for item in eligible if item[0] >= current_index - 60]
        output[row] = (
            float(len(eligible)),
            float(len(recent_20)),
            float(len(recent_60)),
            _mean_return(eligible),
            _mean_return(recent_20),
            _mean_return(recent_60),
            _positive_rate(eligible),
            _positive_rate(recent_20),
            _positive_rate(recent_60),
        )
        history.append((current_index, float(excess_returns[row])))
    return output


def _mean_return(values: list[tuple[int, float]]) -> float:
    return sum(value for _, value in values) / len(values) if values else 0.0


def _positive_rate(values: list[tuple[int, float]]) -> float:
    return (
        sum(value > 0 for _, value in values) / len(values)
        if values
        else 0.0
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embargo-trading-dates", type=int, default=6)
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
