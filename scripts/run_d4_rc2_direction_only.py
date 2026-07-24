"""Run D4-RC2 by removing only the auxiliary return loss from E4."""

from __future__ import annotations

import json
import pickle
from hashlib import sha256
from pathlib import Path

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

ROOT = Path("artifacts/diagnostics/D4_RC2_DIRECTION_ONLY")
CHUNK = Path("artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/tensorized_chunks/chunk-000000.pkl")
SEED = 2026


def main() -> None:
    if not CHUNK.is_file():
        raise FileNotFoundError(f"D4-RC2 requires verified current-schema chunk: {CHUNK}")
    with CHUNK.open("rb") as stream:
        samples = tuple(pickle.load(stream))[:32]
    if len(samples) != 32:
        raise ValueError("D4-RC2 requires exactly 32 samples")

    return_state = fit_return_transform(samples)
    config = DTSeatHGTTrainingConfig(
        model=DTSeatHGTConfig(
            hidden_dim=64,
            num_heads=2,
            num_layers=2,
            dropout=0.3,
            horizons=(1, 3, 5),
        ),
        learning_rate=3e-4,
        epochs=100,
        seeds=(SEED,),
        loss_direction_weight=1.0,
        loss_return_weight=0.0,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        early_stopping_patience=12,
    )
    training = train_dt_seat_hgt(
        samples,
        config,
        ROOT,
        run_id="d4-rc2-direction-only",
        return_transform_state=return_state,
    )
    restored = restore_dt_seat_hgt_seed(
        training.checkpoint_path,
        samples[0],
        SEED,
        return_transform_state=return_state,
    )
    snapshot = prediction_snapshot_for_samples(
        restored.model,
        samples,
        (1, 3, 5),
        return_transform_state=return_state,
    )
    scores = _scores(samples, snapshot, 1, SEED, "micro")
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
        "finite_epochs": all(item["probabilities"]["std"] >= 0 for item in epochs),
        "no_probability_saturation": all(
            item["probabilities"]["saturation_ratio"] < 0.95 for item in epochs
        ),
    }
    conclusion = (
        "auxiliary_return_objective_implicated"
        if gate["auc_gt_0_95"]
        else "auxiliary_return_objective_not_sufficient_core_path_still_suspect"
    )
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "promotion_permitted": False,
        "diagnostic": "D4-RC2 direction-only loss isolation",
        "source_chunk": str(CHUNK),
        "sample_count": len(samples),
        "changed_variable": {
            "name": "loss_return_weight",
            "e4_value": 1.0,
            "rc2_value": 0.0,
        },
        "config": config.to_dict(),
        "return_transform": return_state.to_dict(),
        "checkpoint_path": str(training.checkpoint_path),
        "metrics": metrics.to_dict(),
        "gate": gate,
        "gate_passed": all(gate.values()),
        "conclusion": conclusion,
        "epochs": epochs,
        "scope_note": (
            "This is a fixed-slice training diagnostic only and cannot support "
            "out-of-sample, promotion, or trading claims."
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload["sha256"] = sha256(serialized).hexdigest()
    ROOT.mkdir(parents=True, exist_ok=True)
    result_path = ROOT / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"result": str(result_path), "gate": gate, "conclusion": conclusion},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
