"""Run the D4 micro-overfit gate on a fixed 32-sample, current-schema slice."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from seatflow.evaluation.dt_rolling import _scores
from seatflow.models.seat_hgt import DTSeatHGTConfig
from seatflow.research import evaluate_predictions
from seatflow.return_transforms import fit_return_transform
from seatflow.training.dtseathgt import (
    DTSeatHGTTrainingConfig,
    prediction_snapshot_for_samples,
    restore_dt_seat_hgt_seed,
    train_dt_seat_hgt,
)

ROOT = Path("artifacts/diagnostics/E4_MICRO_OVERFIT_CONTROL_LABELS")
CHUNK = Path("artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/tensorized_chunks/chunk-000000.pkl")


def main() -> None:
    if not CHUNK.is_file():
        raise FileNotFoundError(f"E4 requires verified current-schema chunk: {CHUNK}")
    with CHUNK.open("rb") as stream:
        raw_samples = tuple(pickle.load(stream))[:32]
    samples = _control_label_samples(raw_samples)
    if len(samples) != 32:
        raise ValueError("E4 requires exactly 32 samples")
    state = fit_return_transform(samples)
    config = DTSeatHGTTrainingConfig(
        model=DTSeatHGTConfig(
            hidden_dim=64, num_heads=2, num_layers=2, dropout=0.3, horizons=(1, 3, 5)
        ),
        learning_rate=3e-4,
        epochs=100,
        seeds=(2026,),
        weight_decay=1e-4,
        early_stopping_patience=12,
    )
    training = train_dt_seat_hgt(
        samples,
        config,
        ROOT,
        run_id="e4-micro-overfit",
        return_transform_state=state,
    )
    restored = restore_dt_seat_hgt_seed(
        training.checkpoint_path, samples[0], 2026, return_transform_state=state
    )
    snapshot = prediction_snapshot_for_samples(
        restored.model, samples, (1, 3, 5), return_transform_state=state
    )
    scores = _scores(samples, snapshot, 1, 2026, "micro")
    metrics = evaluate_predictions(
        tuple(float(item["score"]) for item in scores),
        tuple(int(item["label"]) for item in scores),
        tuple(float(item["label_return"]) for item in scores),
    )
    checkpoint = restored.checkpoint
    seed_run = checkpoint["seed_runs"][0]
    epochs = seed_run["epochs"]
    losses = [float(item["total_loss"]) for item in epochs]
    gate = {
        "auc_gt_0_95": metrics.auc is not None and metrics.auc > 0.95,
        "loss_decreased": losses[-1] < losses[0],
        "finite_epochs": all(item["predicted_returns"]["std"] >= 0 for item in epochs),
        "no_output_explosion": all(abs(item["predicted_returns"]["max"]) < 10 for item in epochs),
    }
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "negative_result": True,
        "sample_count": len(samples),
        "label_source": "frozen_control_from_stock_return_1d",
        "config": config.to_dict(),
        "return_transform": state.to_dict(),
        "checkpoint_path": str(training.checkpoint_path),
        "metrics": metrics.to_dict(),
        "gate": gate,
        "gate_passed": all(gate.values()),
        "epochs": epochs,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["sha256"] = sha256(serialized).hexdigest()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"result": str(ROOT / "result.json"), "gate": gate}, sort_keys=True))


def _control_label_samples(samples: tuple[Any, ...]) -> tuple[Any, ...]:
    if not samples:
        return ()
    first = samples[0]
    stock_schema = first.graph.metadata["feature_schema"]["stock"]
    feature_index = stock_schema.index("stock_return_1d")
    values = sorted(
        float(sample.graph.data["stock"].x[sample.target_node_index, feature_index])
        for sample in samples
    )
    threshold = values[len(values) // 2]
    return tuple(
        replace(
            sample,
            labels={
                horizon: int(
                    float(sample.graph.data["stock"].x[sample.target_node_index, feature_index])
                    > threshold
                )
                for horizon in (1, 3, 5)
            },
            returns={
                horizon: (
                    0.02
                    if float(sample.graph.data["stock"].x[sample.target_node_index, feature_index])
                    > threshold
                    else -0.02
                )
                for horizon in (1, 3, 5)
            },
        )
        for sample in samples
    )


if __name__ == "__main__":
    main()
