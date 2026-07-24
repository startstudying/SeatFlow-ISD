"""Run one fail-closed three-class training/evaluation pass per prebuilt V2 fold."""

from __future__ import annotations

import argparse
import json
import pickle
from hashlib import sha256
from pathlib import Path

from seatflow.evaluation.dt_rolling import build_dt_rolling_folds
from seatflow.evaluation.v2_metrics import evaluate_v2_rows, v2_rows_from_prediction_snapshot
from seatflow.models.seat_hgt import DTSeatHGTConfig
from seatflow.training.dtseathgt import (
    DTSeatHGTTrainingConfig,
    prediction_snapshot_for_samples,
    restore_dt_seat_hgt_seed,
    train_dt_seat_hgt,
)

DEFAULT_ROOT = Path("artifacts/diagnostics/R0_V2_TENSOR_DATASET_2024_01_02_22")
DEFAULT_OUT = Path("artifacts/diagnostics/R0_V2_THREE_CLASS_FOLD_SMOKE_2024_01_02_22")


def main() -> None:
    args = _parse_args()
    root = args.dataset_root
    out = args.output_root
    samples = pickle.loads((root / "v2-tensor-dataset.pkl").read_bytes())
    dataset_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    folds = build_dt_rolling_folds(
        samples,
        selection_horizon=1,
        n_folds=3,
        min_train_dates=args.min_train_dates,
    )
    config = DTSeatHGTTrainingConfig(
        model=DTSeatHGTConfig(
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            horizons=(1, 3, 5),
            direction_classes=3,
        ),
        epochs=1,
        seeds=(2026,),
        early_stopping_patience=1,
        scheduler_patience=1,
    )
    rows: list[dict[str, object]] = []
    for fold in folds:
        result = train_dt_seat_hgt(
            fold.train, config, out, run_id=fold.fold_id, validation_samples=fold.test
        )
        restored = restore_dt_seat_hgt_seed(result.checkpoint_path, fold.test[0], 2026)
        snapshot = prediction_snapshot_for_samples(restored.model, fold.test, (1, 3, 5))
        metrics = evaluate_v2_rows(v2_rows_from_prediction_snapshot(fold.test, snapshot, horizon=1))
        rows.append(
            {
                "fold": fold.fold_id,
                "checkpoint": str(result.checkpoint_path),
                "checkpoint_sha256": _file_hash(result.checkpoint_path),
                "train_sample_count": len(fold.train),
                "test_sample_count": len(fold.test),
                "metrics": metrics,
            }
        )
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "experiment": "R0_V2_THREE_CLASS_FOLD_SMOKE",
        "single_changed_variable": args.single_changed_variable,
        "code_sha256": _file_hash(Path(__file__)),
        "dataset_root": str(root),
        "dataset_manifest_sha256": dataset_manifest["sha256"],
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "tensor_schema": "executable_excess_return_v2",
        "min_train_dates": args.min_train_dates,
        "config": config.to_dict(),
        "folds": [fold.to_dict() for fold in folds],
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"folds": len(rows)}))


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fail-closed V2 three-class fold smoke experiment."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=6,
        help="Minimum distinct as-of dates in each training fold.",
    )
    parser.add_argument(
        "--single-changed-variable",
        default="v2_three_class_label_schema",
        help="Audit description of the sole experimental variable relative to the prior run.",
    )
    args = parser.parse_args()
    if args.min_train_dates < 1:
        parser.error("--min-train-dates must be positive")
    return args


if __name__ == "__main__":
    main()
