"""Add point-in-time Tushare daily_basic features to an audited R1 matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

VALUE_FIELDS: tuple[str, ...] = (
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
)
DAILY_BASIC_FEATURE_NAMES: tuple[str, ...] = (
    "daily_basic::available",
    *(f"daily_basic::{field}" for field in VALUE_FIELDS),
    "daily_basic::circ_to_total_mv",
    "daily_basic::float_to_total_share",
    "daily_basic::free_to_float_share",
    *(f"daily_basic::{field}_missing" for field in VALUE_FIELDS),
)


def main() -> None:
    args = _arguments()
    source_manifest = json.loads(
        (args.matrix_root / "manifest.json").read_text(encoding="utf-8")
    )
    source_matrix_path = Path(str(source_manifest["matrix"]))
    if _file_sha256(source_matrix_path) != source_manifest.get("matrix_sha256"):
        raise ValueError("R1 source matrix hash mismatch")
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    backfill_manifest = json.loads(args.backfill_manifest.read_text(encoding="utf-8"))
    if (
        dataset_manifest.get("dataset_sha256")
        != source_manifest.get("source_dataset_sha256")
        or backfill_manifest.get("dataset_manifest_sha256")
        != dataset_manifest.get("sha256")
    ):
        raise ValueError("R1 daily_basic inputs do not share the frozen dataset")
    if backfill_manifest.get("completed_date_count") != len(
        dataset_manifest["source_as_of_dates"]
    ):
        raise ValueError("R1 daily_basic backfill is incomplete")

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    feature_names = [str(value) for value in source["feature_names"]]
    daily_basic, coverage, missingness = _daily_basic_features(
        sample_ids,
        as_of_dates,
        [str(value) for value in dataset_manifest["source_as_of_dates"]],
        args.data_root,
    )
    augmented = np.concatenate((x, daily_basic), axis=1)
    augmented_names = feature_names + list(DAILY_BASIC_FEATURE_NAMES)

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
        feature_names=np.asarray(augmented_names),
    )
    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "label_schema_version": source_manifest["label_schema_version"],
        "source_manifest": str(args.matrix_root / "manifest.json"),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_dataset_sha256": source_manifest["source_dataset_sha256"],
        "daily_basic_backfill_manifest": str(args.backfill_manifest),
        "daily_basic_backfill_sha256": backfill_manifest["sha256"],
        "daily_basic_sample_coverage": coverage,
        "daily_basic_missingness": missingness,
        "sample_count": len(sample_ids),
        "feature_count": len(augmented_names),
        "feature_names": augmented_names,
        "added_feature_block": "point_in_time_daily_basic",
        "added_feature_names": list(DAILY_BASIC_FEATURE_NAMES),
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
                "coverage": coverage,
                "matrix_sha256": payload["matrix_sha256"],
            }
        )
    )


def _daily_basic_features(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    source_calendar: list[str],
    data_root: Path,
) -> tuple[np.ndarray, float, dict[str, int]]:
    if sample_ids.shape != as_of_dates.shape or sample_ids.ndim != 1:
        raise ValueError("R1 daily_basic feature inputs must be aligned vectors")
    rows_by_key: dict[tuple[str, str], int] = {}
    targets_by_date: dict[str, set[str]] = defaultdict(set)
    for row, (sample_id, day) in enumerate(zip(sample_ids, as_of_dates, strict=True)):
        stock_id = str(sample_id).rsplit("::", maxsplit=1)[-1]
        key = str(day), stock_id
        rows_by_key[key] = row
        targets_by_date[str(day)].add(stock_id)

    output = np.zeros(
        (len(sample_ids), len(DAILY_BASIC_FEATURE_NAMES)), dtype=np.float64
    )
    seen: set[tuple[str, str]] = set()
    missingness = {field: 0 for field in VALUE_FIELDS}
    for day in source_calendar:
        path = data_root / "tushare" / "daily_basic" / f"date={day}" / "data.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"daily_basic snapshot is not a list: {path}")
        targets = targets_by_date.get(day, set())
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"daily_basic row is not an object: {path}")
            stock_id = str(value["stock_id"])
            if stock_id not in targets:
                continue
            row = rows_by_key[(day, stock_id)]
            output[row] = _daily_basic_feature_row(value)
            for field in VALUE_FIELDS:
                missingness[field] += int(value.get(field) is None)
            seen.add((day, stock_id))
    coverage = len(seen) / len(rows_by_key)
    return output, coverage, missingness


def _daily_basic_feature_row(value: dict[str, object]) -> tuple[float, ...]:
    raw = [value.get(field) for field in VALUE_FIELDS]
    numbers = [0.0 if item is None else float(item) for item in raw]
    (
        turnover_rate,
        turnover_rate_f,
        volume_ratio,
        total_share,
        float_share,
        free_share,
        total_mv,
        circ_mv,
    ) = numbers
    return (
        1.0,
        turnover_rate,
        turnover_rate_f,
        volume_ratio,
        total_share,
        float_share,
        free_share,
        total_mv,
        circ_mv,
        circ_mv / total_mv if total_mv > 0 else 0.0,
        float_share / total_share if total_share > 0 else 0.0,
        free_share / float_share if float_share > 0 else 0.0,
        *(1.0 if item is None else 0.0 for item in raw),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--backfill-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
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
