"""Add point-in-time SW2021 industry features to an audited R1 daily_basic matrix."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

INDUSTRY_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "industry::available",
    "industry::membership_age_days",
)


def main() -> None:
    args = _arguments()
    source_manifest_path = args.matrix_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("added_feature_block") != "point_in_time_daily_basic":
        raise ValueError("R1 industry features require the audited daily_basic matrix")
    source_matrix_path = Path(str(source_manifest["matrix"]))
    if _file_sha256(source_matrix_path) != source_manifest.get("matrix_sha256"):
        raise ValueError("R1 source matrix hash mismatch")

    backfill_manifest = json.loads(args.backfill_manifest.read_text(encoding="utf-8"))
    rows, canonical_sha256 = _load_backfill_rows(args.backfill_manifest, backfill_manifest)

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    source_feature_names = [str(value) for value in source["feature_names"]]
    industry, feature_names, coverage = _industry_features(sample_ids, as_of_dates, rows)
    augmented_names = source_feature_names + list(feature_names)
    augmented = np.concatenate((x, industry), axis=1)

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
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_dataset_sha256": source_manifest["source_dataset_sha256"],
        "source_matrix_sha256": source_manifest["matrix_sha256"],
        "industry_backfill_manifest": str(args.backfill_manifest),
        "industry_backfill_sha256": backfill_manifest.get("sha256"),
        "industry_membership_canonical_sha256": canonical_sha256,
        "industry_sample_coverage": coverage,
        "sample_count": len(sample_ids),
        "feature_count": len(augmented_names),
        "feature_names": augmented_names,
        "added_feature_block": "point_in_time_sw2021_industry",
        "added_feature_names": list(feature_names),
        "sample_ids_sha256": source_manifest["sample_ids_sha256"],
        "matrix": str(matrix_path),
        "matrix_sha256": _file_sha256(matrix_path),
        "code_sha256": _file_sha256(Path(__file__)),
    }
    if "event_amount_semantics" in source_manifest:
        payload["event_amount_semantics"] = source_manifest["event_amount_semantics"]
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


def _industry_features(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    rows: list[dict[str, object]],
) -> tuple[np.ndarray, tuple[str, ...], float]:
    if sample_ids.shape != as_of_dates.shape or sample_ids.ndim != 1:
        raise ValueError("R1 industry feature inputs must be aligned vectors")
    l1_codes = _sw2021_l1_codes(rows)
    feature_names = INDUSTRY_BASE_FEATURE_NAMES + tuple(
        f"industry::sw2021_l1::{code}" for code in l1_codes
    )
    output = np.zeros((len(sample_ids), len(feature_names)), dtype=np.float64)
    memberships = _index_memberships_by_stock(rows)
    assigned = 0
    for row, (sample_id, as_of_date) in enumerate(zip(sample_ids, as_of_dates, strict=True)):
        stock_id = str(sample_id).rsplit("::", maxsplit=1)[-1]
        membership = _active_membership(memberships.get(stock_id, ()), str(as_of_date))
        if membership is None:
            continue
        assigned += 1
        output[row, 0] = 1.0
        output[row, 1] = float(_date_to_ordinal(str(as_of_date)) - membership[0])
        l1_code = membership[2]
        if l1_code in l1_codes:
            output[row, 2 + l1_codes.index(l1_code)] = 1.0
    coverage = assigned / len(sample_ids) if len(sample_ids) else 0.0
    return output, feature_names, coverage


def _index_memberships_by_stock(
    rows: list[dict[str, object]],
) -> dict[str, tuple[tuple[int, int | None, str, str, str, bool], ...]]:
    memberships: dict[str, list[tuple[int, int | None, str, str, str, bool]]] = {}
    for row in rows:
        stock_id = _stock_id(row)
        in_date = _date_to_ordinal(_required_string(row, "in_date"))
        out_date_value = row.get("out_date")
        out_date = (
            None
            if out_date_value in (None, "")
            else _date_to_ordinal(str(out_date_value))
        )
        if out_date is not None and out_date <= in_date:
            raise ValueError(f"industry membership has non-positive interval: {stock_id}")
        memberships.setdefault(stock_id, []).append(
            (
                in_date,
                out_date,
                _required_string(row, "l1_code"),
                _required_string(row, "l2_code"),
                _required_string(row, "l3_code"),
                _is_current_membership(row.get("is_new")),
            )
        )
    return {
        stock_id: tuple(sorted(values, key=lambda item: (item[0], item[2], item[3], item[4])))
        for stock_id, values in memberships.items()
    }


def _active_membership(
    memberships: tuple[tuple[int, int | None, str, str, str, bool], ...],
    as_of_date: str,
) -> tuple[int, int | None, str, str, str, bool] | None:
    as_of = _date_to_ordinal(as_of_date)
    active = [
        membership
        for membership in memberships
        if membership[0] <= as_of and (membership[1] is None or as_of < membership[1])
    ]
    return max(active, key=lambda item: (item[0], item[2], item[3], item[4])) if active else None


def _sw2021_l1_codes(rows: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({_required_string(row, "l1_code") for row in rows}))


def _load_backfill_rows(
    manifest_path: Path, manifest: dict[str, object]
) -> tuple[list[dict[str, object]], str]:
    paths = _snapshot_paths(manifest_path, manifest)
    rows: list[dict[str, object]] = []
    for path in paths:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"index_member_all snapshot is not a list: {path}")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"index_member_all row is not an object: {path}")
            rows.append(value)
    return rows, _canonical_rows_sha256(rows)


def _snapshot_paths(manifest_path: Path, manifest: dict[str, object]) -> list[Path]:
    raw_paths = manifest.get("snapshots", manifest.get("snapshot_paths"))
    if raw_paths is None and "snapshot" in manifest:
        raw_paths = [manifest["snapshot"]]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("industry backfill manifest requires snapshot paths")
    paths: list[Path] = []
    for raw_path in raw_paths:
        if isinstance(raw_path, dict):
            raw_path = raw_path.get("data_path", raw_path.get("path"))
        if not isinstance(raw_path, str):
            raise ValueError("industry backfill snapshot path is not a string")
        path = Path(raw_path)
        if not path.is_absolute() and not path.exists():
            path = manifest_path.parent / path
        paths.append(path)
    return paths


def _canonical_rows_sha256(rows: list[dict[str, object]]) -> str:
    canonical = [
        {
            "stock_id": _stock_id(row),
            "l1_code": _required_string(row, "l1_code"),
            "l2_code": _required_string(row, "l2_code"),
            "l3_code": _required_string(row, "l3_code"),
            "in_date": _required_string(row, "in_date"),
            "out_date": None if row.get("out_date") in (None, "") else str(row["out_date"]),
            "is_new": _is_current_membership(row.get("is_new")),
        }
        for row in rows
    ]
    canonical.sort(
        key=lambda row: (
            row["stock_id"],
            row["in_date"],
            row["out_date"] or "",
            row["l1_code"],
            row["l2_code"],
            row["l3_code"],
            row["is_new"],
        )
    )
    return _hash_json(canonical)


def _required_string(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        raise ValueError(f"industry membership row is missing {field}")
    return str(value)


def _stock_id(row: dict[str, object]) -> str:
    for field in ("stock_id", "ts_code", "con_code"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError("industry membership row is missing stock_id")


def _is_current_membership(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "Y"


def _date_to_ordinal(value: str) -> int:
    parts = value.split("-")
    if len(parts) != 3:
        raise ValueError(f"invalid ISO date: {value}")
    year, month, day = (int(part) for part in parts)
    return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D").astype(int).item()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--backfill-manifest", type=Path, required=True)
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
