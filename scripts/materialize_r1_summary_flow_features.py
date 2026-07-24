"""Add audited Dragon-Tiger summary flow features to an R1 matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "summary_flow::available",
    "summary_flow::row_count",
    "summary_flow::buy_amount",
    "summary_flow::sell_amount",
    "summary_flow::net_amount",
    "summary_flow::gross_amount",
    "summary_flow::buy_to_daily_amount",
    "summary_flow::sell_to_daily_amount",
    "summary_flow::net_to_daily_amount",
    "summary_flow::net_to_gross",
    "summary_flow::institution_buy_share",
    "summary_flow::institution_sell_share",
    "summary_flow::institution_gross_share",
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
    if (
        dataset_manifest.get("dataset_sha256")
        != source_manifest.get("source_dataset_sha256")
    ):
        raise ValueError("R1 summary features require the matching dataset manifest")

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    feature_names = [str(value) for value in source["feature_names"]]
    daily_amount_index = feature_names.index("market::daily_amount_cny")
    institution_buy_index = feature_names.index("event_profile::0")
    institution_sell_index = feature_names.index("event_profile::1")
    summary, snapshot_sha256, coverage = _summary_features(
        sample_ids,
        as_of_dates,
        x[:, daily_amount_index],
        x[:, institution_buy_index],
        x[:, institution_sell_index],
        [str(value) for value in dataset_manifest["source_as_of_dates"]],
        args.data_root,
    )
    augmented = np.concatenate((x, summary), axis=1)
    augmented_names = feature_names + list(SUMMARY_FEATURE_NAMES)

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
        "source_summary_root": str(args.data_root),
        "source_summary_snapshot_sha256": snapshot_sha256,
        "summary_sample_coverage": coverage,
        "sample_count": len(sample_ids),
        "feature_count": len(augmented_names),
        "feature_names": augmented_names,
        "added_feature_block": "dragon_tiger_summary_flow",
        "added_feature_names": list(SUMMARY_FEATURE_NAMES),
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


def _summary_features(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    daily_amounts: np.ndarray,
    institution_buy: np.ndarray,
    institution_sell: np.ndarray,
    source_calendar: list[str],
    data_root: Path,
) -> tuple[np.ndarray, str, float]:
    aligned = (
        sample_ids.shape
        == as_of_dates.shape
        == daily_amounts.shape
        == institution_buy.shape
        == institution_sell.shape
    )
    if not aligned or sample_ids.ndim != 1:
        raise ValueError("R1 summary feature inputs must be aligned vectors")
    rows_by_key: dict[tuple[str, str], int] = {}
    targets_by_date: dict[str, set[str]] = defaultdict(set)
    for row, (sample_id, day) in enumerate(zip(sample_ids, as_of_dates, strict=True)):
        stock_id = str(sample_id).rsplit("::", maxsplit=1)[-1]
        key = str(day), stock_id
        rows_by_key[key] = row
        targets_by_date[str(day)].add(stock_id)

    output = np.zeros((len(sample_ids), len(SUMMARY_FEATURE_NAMES)), dtype=np.float64)
    seen: set[tuple[str, str]] = set()
    snapshot_digest = sha256()
    for day in source_calendar:
        path = data_root / "dragon_tiger_summary" / f"date={day}" / "data.json"
        raw = path.read_bytes()
        file_digest = sha256(raw).hexdigest()
        snapshot_digest.update(f"{day}:{file_digest}\n".encode())
        values = json.loads(raw)
        if not isinstance(values, list):
            raise ValueError(f"summary snapshot is not a list: {path}")
        aggregated: dict[str, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0]
        )
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"summary row is not an object: {path}")
            stock_id = str(value["stock_id"])
            aggregate = aggregated[stock_id]
            aggregate[0] += 1.0
            aggregate[1] += float(value["buy_amount"])
            aggregate[2] += float(value["sell_amount"])
            aggregate[3] += float(value["net_amount"])
        for stock_id in targets_by_date.get(day, set()):
            aggregate = aggregated.get(stock_id)
            if aggregate is None:
                continue
            row = rows_by_key[(day, stock_id)]
            output[row] = _summary_feature_row(
                row_count=aggregate[0],
                summary_buy=aggregate[1],
                summary_sell=aggregate[2],
                summary_net=aggregate[3],
                daily_amount=float(daily_amounts[row]),
                institution_buy=float(institution_buy[row]),
                institution_sell=float(institution_sell[row]),
            )
            seen.add((day, stock_id))
    coverage = len(seen) / len(rows_by_key)
    return output, snapshot_digest.hexdigest(), coverage


def _summary_feature_row(
    *,
    row_count: float,
    summary_buy: float,
    summary_sell: float,
    summary_net: float,
    daily_amount: float,
    institution_buy: float,
    institution_sell: float,
) -> tuple[float, ...]:
    if daily_amount <= 0:
        raise ValueError("R1 summary normalization requires positive daily amount")
    gross = summary_buy + summary_sell
    institution_gross = institution_buy + institution_sell
    return (
        1.0,
        row_count,
        summary_buy,
        summary_sell,
        summary_net,
        gross,
        summary_buy / daily_amount,
        summary_sell / daily_amount,
        summary_net / daily_amount,
        summary_net / gross if gross > 0 else 0.0,
        institution_buy / summary_buy if summary_buy > 0 else 0.0,
        institution_sell / summary_sell if summary_sell > 0 else 0.0,
        institution_gross / gross if gross > 0 else 0.0,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
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
