"""Run the independent D4 feature-representability root-cause probe.

This diagnostic deliberately avoids DT-SeatHGT and its training loop.  It asks
whether the failed E4 slice can be memorized from the current target-stock
features by a small, high-capacity MLP.  A synthetic feature-derived label is
trained with the identical probe as a positive control.
"""

from __future__ import annotations

import json
import pickle
import random
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any

from seatflow.research import evaluate_predictions

CHUNK = Path("artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/tensorized_chunks/chunk-000000.pkl")
ROOT = Path("artifacts/diagnostics/D4_ROOT_CAUSE_FEATURE_PROBE")
SEEDS = (2026, 2027, 2028)
HORIZON = 1
SAMPLE_COUNT = 32


def _target_features(samples: tuple[Any, ...], torch: Any) -> Any:
    return torch.stack(
        [
            sample.graph.data["stock"]
            .x[sample.target_node_index]
            .detach()
            .to(dtype=torch.float32, device="cpu")
            for sample in samples
        ]
    )


def _standardize(features: Any) -> Any:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def _feature_fingerprint(row: Any) -> str:
    payload = row.detach().contiguous().cpu().numpy().tobytes()
    return sha256(payload).hexdigest()


def _collision_audit(features: Any, labels: list[int]) -> dict[str, object]:
    grouped: dict[str, set[int]] = {}
    for row, label in zip(features, labels, strict=True):
        grouped.setdefault(_feature_fingerprint(row), set()).add(label)
    conflicts = sorted(key for key, values in grouped.items() if len(values) > 1)
    return {
        "unique_feature_rows": len(grouped),
        "duplicate_feature_rows": len(labels) - len(grouped),
        "conflicting_exact_feature_rows": len(conflicts),
        "conflict_fingerprints": conflicts,
    }


def _synthetic_labels(features: Any) -> list[int]:
    weights = features.new_tensor(
        [1.0 if index % 2 == 0 else -0.75 for index in range(features.size(1))]
    )
    scores = (features * weights).sum(dim=1).tolist()
    threshold = median(scores)
    return [int(score > threshold) for score in scores]


def _train_probe(features: Any, labels: list[int], seed: int, torch: Any) -> dict[str, object]:
    from torch import nn

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = nn.Sequential(
        nn.Linear(int(features.size(1)), 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    target = torch.tensor(labels, dtype=torch.float32)
    initial_loss: float | None = None
    final_loss = float("nan")
    epochs_run = 0
    for epoch in range(1, 3001):
        model.train()
        optimizer.zero_grad()
        logits = model(features).squeeze(-1)
        loss = criterion(logits, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("D4 feature probe produced non-finite loss")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        initial_loss = final_loss if initial_loss is None else initial_loss
        epochs_run = epoch
        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(model(features).squeeze(-1)).tolist()
            metrics = evaluate_predictions(
                tuple(float(value) for value in probabilities),
                tuple(labels),
                tuple(float(value) for value in labels),
            )
            if metrics.auc is not None and metrics.auc > 0.999 and final_loss < 0.03:
                break
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features).squeeze(-1)).tolist()
    metrics = evaluate_predictions(
        tuple(float(value) for value in probabilities),
        tuple(labels),
        tuple(float(value) for value in labels),
    )
    return {
        "seed": seed,
        "epochs": epochs_run,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "metrics": metrics.to_dict(),
        "probability_min": min(probabilities),
        "probability_max": max(probabilities),
    }


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("D4 feature probe requires PyTorch") from error
    if not CHUNK.is_file():
        raise FileNotFoundError(f"D4 feature probe requires {CHUNK}")
    with CHUNK.open("rb") as stream:
        samples = tuple(pickle.load(stream))[:SAMPLE_COUNT]
    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"D4 feature probe requires exactly {SAMPLE_COUNT} samples")

    raw_features = _target_features(samples, torch)
    features = _standardize(raw_features)
    real_labels = [int(sample.labels[HORIZON]) for sample in samples]
    synthetic_labels = _synthetic_labels(features)
    collision_audit = _collision_audit(raw_features, real_labels)
    real_runs = [_train_probe(features, real_labels, seed, torch) for seed in SEEDS]
    control_runs = [_train_probe(features, synthetic_labels, seed, torch) for seed in SEEDS]

    real_passes = sum(float(run["metrics"]["auc"]) > 0.95 for run in real_runs)
    control_passes = sum(float(run["metrics"]["auc"]) > 0.95 for run in control_runs)
    gate = {
        "positive_control_passed": control_passes == len(SEEDS),
        "no_exact_feature_label_conflicts": collision_audit["conflicting_exact_feature_rows"] == 0,
        "real_label_probe_auc_gt_0_95_at_least_two_seeds": real_passes >= 2,
    }
    if not gate["positive_control_passed"]:
        conclusion = "probe_invalid_training_or_implementation_defect"
    elif gate["real_label_probe_auc_gt_0_95_at_least_two_seeds"]:
        conclusion = "labels_representable_on_micro_slice_model_or_training_path_suspect"
    else:
        conclusion = "current_target_features_insufficient_on_micro_slice"
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "promotion_permitted": False,
        "diagnostic": "D4 independent target-feature representability probe",
        "source_chunk": str(CHUNK),
        "sample_ids": [sample.sample_id for sample in samples],
        "sample_count": len(samples),
        "horizon": HORIZON,
        "feature_width": int(features.size(1)),
        "real_positive_count": sum(real_labels),
        "synthetic_positive_count": sum(synthetic_labels),
        "collision_audit": collision_audit,
        "real_label_runs": real_runs,
        "positive_control_runs": control_runs,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "conclusion": conclusion,
        "scope_note": (
            "This is a training-slice memorization diagnostic only. It does not establish "
            "out-of-sample signal, model validity, or trading readiness."
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
