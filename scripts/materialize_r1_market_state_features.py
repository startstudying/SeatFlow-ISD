"""Add point-in-time broad-market state features to an audited R1 matrix."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import isfinite, log1p
from pathlib import Path

import numpy as np

MARKET_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "market_state::equal_weight_return_1d",
    "market_state::median_return_1d",
    "market_state::positive_breadth",
    "market_state::cross_sectional_volatility",
    "market_state::log_total_amount_cny",
    "market_state::equal_weight_momentum_5d",
    "market_state::equal_weight_momentum_20d",
    "market_state::equal_weight_volatility_20d",
)


def main() -> None:
    args = _arguments()
    source_manifest_path = args.matrix_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("added_feature_block") != "point_in_time_sw2021_industry":
        raise ValueError("R1 market state requires the audited industry matrix")
    source_matrix_path = Path(str(source_manifest["matrix"]))
    if _file_sha256(source_matrix_path) != source_manifest.get("matrix_sha256"):
        raise ValueError("R1 market-state source matrix hash mismatch")

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    source_feature_names = [str(value) for value in source["feature_names"]]
    unique_dates = sorted({str(value) for value in as_of_dates})
    daily_rows = {
        day: _load_daily_bars(args.daily_bar_root, day) for day in unique_dates
    }
    state_by_date = _market_state_by_date(daily_rows)
    state = np.asarray(
        [state_by_date[str(day)] for day in as_of_dates],
        dtype=np.float64,
    )
    if state.shape != (len(sample_ids), len(MARKET_STATE_FEATURE_NAMES)):
        raise ValueError("R1 market-state feature matrix is misaligned")
    if not np.isfinite(state).all():
        raise ValueError("R1 market-state features contain non-finite values")

    augmented_names = source_feature_names + list(MARKET_STATE_FEATURE_NAMES)
    augmented = np.concatenate((x, state), axis=1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_root / "matrix.npz"
    np.savez_compressed(
        matrix_path,
        x=augmented,
        sample_ids=sample_ids,
        as_of_dates=as_of_dates,
        labels=labels,
        excess_returns=excess_returns,
        feature_names=np.asarray(augmented_names),
    )
    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "label_schema_version": source_manifest["label_schema_version"],
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_dataset_sha256": source_manifest["source_dataset_sha256"],
        "source_matrix_sha256": source_manifest["matrix_sha256"],
        "industry_sample_coverage": source_manifest["industry_sample_coverage"],
        "daily_bar_root": str(args.daily_bar_root),
        "market_state_date_coverage": 1.0,
        "sample_count": len(sample_ids),
        "feature_count": len(augmented_names),
        "feature_names": augmented_names,
        "added_feature_block": "point_in_time_market_state",
        "added_feature_names": list(MARKET_STATE_FEATURE_NAMES),
        "sample_ids_sha256": source_manifest["sample_ids_sha256"],
        "matrix": str(matrix_path),
        "matrix_sha256": _file_sha256(matrix_path),
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "samples": payload["sample_count"],
                "features": payload["feature_count"],
                "date_coverage": payload["market_state_date_coverage"],
                "matrix_sha256": payload["matrix_sha256"],
            }
        )
    )


def _load_daily_bars(root: Path, day: str) -> list[dict[str, object]]:
    path = root / f"date={day}" / "data.json"
    if not path.is_file():
        raise ValueError(f"R1 market-state daily bars are missing {day}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"R1 market-state daily bars are empty {day}")
    if not all(isinstance(row, dict) for row in raw):
        raise ValueError(f"R1 market-state daily bars are invalid {day}")
    return raw


def _market_state_by_date(
    daily_rows: dict[str, list[dict[str, object]]],
) -> dict[str, tuple[float, ...]]:
    dates = sorted(daily_rows)
    daily_base: list[tuple[float, float, float, float, float]] = []
    for day in dates:
        returns: list[float] = []
        total_amount_cny = 0.0
        for row in daily_rows[day]:
            close = float(row["close_price"])
            pre_close = float(row["pre_close"])
            amount = float(row["amount"])
            if pre_close <= 0 or amount < 0:
                raise ValueError(f"invalid R1 market-state daily bar on {day}")
            value = close / pre_close - 1.0
            if not isfinite(value):
                raise ValueError(f"non-finite R1 market return on {day}")
            returns.append(value)
            total_amount_cny += amount * 1000.0
        values = np.asarray(returns, dtype=np.float64)
        daily_base.append(
            (
                float(np.mean(values)),
                float(np.median(values)),
                float(np.mean(values > 0)),
                float(np.std(values)),
                log1p(total_amount_cny),
            )
        )

    output: dict[str, tuple[float, ...]] = {}
    equal_weight = np.asarray([row[0] for row in daily_base], dtype=np.float64)
    for index, day in enumerate(dates):
        trailing_5 = equal_weight[max(0, index - 4) : index + 1]
        trailing_20 = equal_weight[max(0, index - 19) : index + 1]
        output[day] = daily_base[index] + (
            float(np.sum(trailing_5)),
            float(np.sum(trailing_20)),
            float(np.std(trailing_20)),
        )
    return output


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--daily-bar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
