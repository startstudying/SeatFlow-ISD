"""Build a manifest-bound V2 tensor chunk from verified snapshots and labels."""

from __future__ import annotations

import argparse
import json
import pickle
from hashlib import sha256
from pathlib import Path

from seatflow.datasets.pyg import tensorize_window_samples
from seatflow.datasets.snapshots import load_verified_historical_snapshots
from seatflow.datasets.v2_labels import load_v2_label_artifact
from seatflow.datasets.windows import DatasetManifest, build_window_samples


def main() -> None:
    args = _arguments()
    label_artifact = load_v2_label_artifact(args.label_artifact)
    labels_by_key = label_artifact.labels_by_key()
    as_of_dates = tuple(sorted({label.as_of_date for label in label_artifact.labels}))
    if len(as_of_dates) != 1:
        raise ValueError("a V2 tensor chunk must contain exactly one as-of date")
    as_of_date = as_of_dates[0]
    label_calendar = tuple(
        sorted(
            {
                item
                for label in label_artifact.labels
                if label.disposition == "eligible"
                for item in (label.as_of_date, label.label_start_date, label.label_end_date)
            }
        )
    )
    target_stock_ids = tuple(
        sorted(stock_id for (label_as_of, stock_id) in labels_by_key if label_as_of == as_of_date)
    )
    cutoff = label_artifact.labels[0].cutoff_time
    if label_calendar:
        snapshots = load_verified_historical_snapshots(
            args.data_root, provider=args.provider, trading_calendar=label_calendar
        )
        windows = build_window_samples(
            snapshots.seat_events,
            snapshots.bars,
            as_of_dates=(as_of_date,),
            target_stock_ids=target_stock_ids,
            horizons=(1, 3, 5),
            cutoff_for_date=lambda _: cutoff,
            decay_lambda=args.decay_lambda,
            max_edge_age_days=args.max_edge_age_days,
            trading_calendar=snapshots.trading_calendar,
            v2_labels_by_key=labels_by_key,
        )
        snapshot_source_hash = snapshots.source_hash
    else:
        windows = ()
        snapshot_source_hash = _hash_json(
            {
                "label_artifact_sha256": label_artifact.artifact_sha256,
                "reason": "no_eligible_v2_labels",
            }
        )
    tensorized = tensorize_window_samples(windows)
    if any(sample.label_schema_version != "executable_excess_return_v2" for sample in tensorized):
        raise ValueError("V2 tensor chunk contains a non-V2 label schema")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    chunk = output / "v2-tensorized-samples.pkl"
    with chunk.open("wb") as stream:
        pickle.dump(tensorized, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = DatasetManifest(
        samples=windows,
        split={"train": [], "validation": [], "test": [sample.sample_id for sample in windows]},
        tensor_shapes={"sample_count": len(tensorized)},
        missingness={"ineligible_v2_labels_excluded": len(target_stock_ids) - len(tensorized)},
    ).to_dict()
    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "label_schema_version": "executable_excess_return_v2",
        "label_artifact_path": str(label_artifact.path),
        "label_artifact_sha256": label_artifact.artifact_sha256,
        "snapshot_source_hash": snapshot_source_hash,
        "tensor_chunk": str(chunk),
        "tensor_chunk_sha256": _file_sha256(chunk),
        "tensor_sample_count": len(tensorized),
        "dataset_manifest": manifest,
    }
    payload["sha256"] = _hash_json(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "tensor_samples": len(tensorized)}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--provider", default="tushare")
    parser.add_argument("--label-artifact", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--decay-lambda", type=float, default=0.1)
    parser.add_argument("--max-edge-age-days", type=int, default=20)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
