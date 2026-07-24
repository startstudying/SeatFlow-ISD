"""Add audited daily-liquidity and flow-normalization features to an R1 matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from hashlib import sha256
from pathlib import Path

import numpy as np

MARKET_FEATURE_NAMES: tuple[str, ...] = (
    "market::daily_amount_cny",
    "market::daily_volume_hand",
    "market::daily_amount_ratio_20d",
    "flow_normalized::institution_buy_to_daily_amount",
    "flow_normalized::institution_sell_to_daily_amount",
    "flow_normalized::institution_net_to_daily_amount",
    "flow_normalized::institution_gross_to_daily_amount",
    "flow_normalized::institution_net_to_gross",
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
        raise ValueError("R1 market features require the matching dataset manifest")
    source_calendar = [str(value) for value in dataset_manifest["source_as_of_dates"]]

    source = np.load(source_matrix_path)
    x = np.asarray(source["x"], dtype=np.float64)
    sample_ids = np.asarray(source["sample_ids"])
    as_of_dates = np.asarray(source["as_of_dates"])
    labels = np.asarray(source["labels"], dtype=np.int8)
    excess_returns = np.asarray(source["excess_returns"], dtype=np.float64)
    feature_names = [str(value) for value in source["feature_names"]]
    buy_index = feature_names.index("event_profile::0")
    sell_index = feature_names.index("event_profile::1")
    net_index = feature_names.index("event_profile::2")
    market, source_snapshot_sha256 = _market_features(
        sample_ids,
        as_of_dates,
        x[:, buy_index],
        x[:, sell_index],
        x[:, net_index],
        source_calendar,
        args.data_root,
    )
    augmented = np.concatenate((x, market), axis=1)
    augmented_names = feature_names + list(MARKET_FEATURE_NAMES)

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
        "source_daily_bar_root": str(args.data_root),
        "source_daily_bar_snapshot_sha256": source_snapshot_sha256,
        "sample_count": len(sample_ids),
        "feature_count": len(augmented_names),
        "feature_names": augmented_names,
        "added_feature_block": "daily_amount_normalized_institution_flow",
        "added_feature_names": list(MARKET_FEATURE_NAMES),
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
                "matrix_sha256": payload["matrix_sha256"],
            }
        )
    )


def _market_features(
    sample_ids: np.ndarray,
    as_of_dates: np.ndarray,
    institution_buy: np.ndarray,
    institution_sell: np.ndarray,
    institution_net: np.ndarray,
    source_calendar: list[str],
    data_root: Path,
) -> tuple[np.ndarray, str]:
    aligned = (
        sample_ids.shape
        == as_of_dates.shape
        == institution_buy.shape
        == institution_sell.shape
        == institution_net.shape
    )
    if not aligned or sample_ids.ndim != 1:
        raise ValueError("R1 market feature inputs must be aligned vectors")
    rows_by_key: dict[tuple[str, str], int] = {}
    targets_by_date: dict[str, set[str]] = defaultdict(set)
    for row, (sample_id, day) in enumerate(zip(sample_ids, as_of_dates, strict=True)):
        stock_id = str(sample_id).rsplit("::", maxsplit=1)[-1]
        key = str(day), stock_id
        if key in rows_by_key:
            raise ValueError(f"duplicate R1 sample key: {key}")
        rows_by_key[key] = row
        targets_by_date[str(day)].add(stock_id)

    output = np.zeros((len(sample_ids), len(MARKET_FEATURE_NAMES)), dtype=np.float64)
    amount_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
    seen: set[tuple[str, str]] = set()
    snapshot_digest = sha256()
    for day in source_calendar:
        path = data_root / "daily_bar" / f"date={day}" / "data.json"
        raw = path.read_bytes()
        file_digest = sha256(raw).hexdigest()
        snapshot_digest.update(f"{day}:{file_digest}\n".encode())
        values = json.loads(raw)
        if not isinstance(values, list):
            raise ValueError(f"daily bar snapshot is not a list: {path}")
        targets = targets_by_date.get(day, set())
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"daily bar row is not an object: {path}")
            stock_id = str(value["stock_id"])
            if value.get("amount_unit") != "thousand_cny":
                raise ValueError(f"unexpected daily amount unit: {path}:{stock_id}")
            amount_cny = float(value["amount"]) * 1000.0
            volume_hand = float(value["volume"])
            history = amount_history[stock_id]
            history.append(amount_cny)
            if stock_id not in targets:
                continue
            row = rows_by_key[(day, stock_id)]
            output[row] = _normalized_market_row(
                amount_cny,
                volume_hand,
                tuple(history),
                float(institution_buy[row]),
                float(institution_sell[row]),
                float(institution_net[row]),
            )
            seen.add((day, stock_id))
    missing = set(rows_by_key) - seen
    if missing:
        example = sorted(missing)[0]
        raise ValueError(f"daily bar snapshots are missing R1 sample {example}")
    return output, snapshot_digest.hexdigest()


def _normalized_market_row(
    amount_cny: float,
    volume_hand: float,
    amount_history: tuple[float, ...],
    institution_buy: float,
    institution_sell: float,
    institution_net: float,
) -> tuple[float, ...]:
    if amount_cny <= 0 or not amount_history:
        raise ValueError("R1 market normalization requires positive daily amount")
    mean_amount = sum(amount_history) / len(amount_history)
    gross = institution_buy + institution_sell
    return (
        amount_cny,
        volume_hand,
        amount_cny / mean_amount,
        institution_buy / amount_cny,
        institution_sell / amount_cny,
        institution_net / amount_cny,
        gross / amount_cny,
        institution_net / gross if gross > 0 else 0.0,
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
