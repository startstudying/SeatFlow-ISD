from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from math import exp, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast
from uuid import uuid4

from seatflow.datasets.pyg import (
    EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
    LEGACY_BINARY_LABEL_SCHEMA,
    TensorizedWindowSample,
)
from seatflow.models.seat_hgt import (
    DTSeatHGTConfig,
    PyGModelUnavailableError,
    build_dt_seat_hgt,
)
from seatflow.return_transforms import ReturnTransformState


class FeatureTransformEvidence(Protocol):
    def fingerprint(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class DTSeatHGTTrainingConfig:
    model: DTSeatHGTConfig
    learning_rate: float = 0.001
    epochs: int = 3
    seeds: tuple[int, ...] = (0,)
    loss_direction_weight: float = 1.0
    loss_return_weight: float = 1.0
    edge_ablation_scale: float = 0.0
    gradient_clip_norm: float | None = 1.0
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    scheduler_patience: int = 4

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique for per-seed checkpoint restore")
        if self.loss_direction_weight < 0 or self.loss_return_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.loss_direction_weight == 0 and self.loss_return_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when set")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stopping_patience < 1 or self.scheduler_patience < 1:
            raise ValueError("training patience values must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        payload["seeds"] = list(self.seeds)
        return payload


@dataclass(frozen=True)
class DTSeatHGTTrainingResult:
    run_id: str
    status: str
    model_valid: bool
    checkpoint_path: Path
    hashes: dict[str, str]
    metrics: dict[str, object]


@dataclass(frozen=True)
class RestoredDTSeatHGT:
    seed: int
    model: Any
    optimizer: Any
    checkpoint: dict[str, object]


def train_dt_seat_hgt(
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    output_root: str | Path,
    run_id: str | None = None,
    *,
    resume_existing: bool = False,
    feature_transform_state: FeatureTransformEvidence | None = None,
    feature_transform_diagnostics: dict[str, object] | None = None,
    return_transform_state: ReturnTransformState | None = None,
    return_transform_diagnostics: dict[str, object] | None = None,
    validation_samples: Sequence[TensorizedWindowSample] = (),
) -> DTSeatHGTTrainingResult:
    _require_non_empty_samples(samples)
    _validate_samples(samples, config.model)
    if validation_samples:
        _validate_samples(validation_samples, config.model)
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:
        raise PyGModelUnavailableError(
            "DTSeatHGT training requires PyTorch and torch-geometric. "
            "Install the ml extra before training."
        ) from error

    effective_run_id = run_id or f"dtseathgt-{uuid4().hex}"
    hashes = _hashes(samples, config, validation_samples)
    if feature_transform_state is not None:
        hashes["feature_transform"] = feature_transform_state.fingerprint()
    if return_transform_state is not None:
        return_transform_state.validate()
        hashes["return_transform"] = return_transform_state.fingerprint()
    checkpoint_path = Path(output_root) / effective_run_id / "dtseathgt.pt"
    progress_path = checkpoint_path.with_name("dtseathgt.in_progress.pt")
    if resume_existing and checkpoint_path.exists():
        return _verified_existing_result(
            torch,
            checkpoint_path,
            effective_run_id,
            samples,
            config,
            hashes,
        )
    progress = (
        _load_training_progress(torch, progress_path, effective_run_id, samples, config, hashes)
        if resume_existing and progress_path.exists()
        else None
    )
    seed_runs: list[dict[str, object]] = (
        list(cast(list[dict[str, object]], progress["seed_runs"])) if progress else []
    )
    state_by_seed: dict[str, dict[str, object]] = (
        dict(cast(dict[str, dict[str, object]], progress["state_by_seed"])) if progress else {}
    )
    for seed in config.seeds:
        if str(seed) in state_by_seed:
            continue
        _set_seed(seed, torch)
        model = build_dt_seat_hgt(config.model, samples[0].graph)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=config.scheduler_patience, factor=0.5
        )
        classification_criterion = (
            nn.BCEWithLogitsLoss()
            if config.model.direction_classes == 2
            else nn.CrossEntropyLoss()
        )
        huber = nn.HuberLoss()
        resumed_seed = _progress_for_seed(progress, seed)
        losses: list[float] = []
        epoch_metrics: list[dict[str, object]] = []
        best_model_state: dict[str, Any] | None = None
        best_epoch = 0
        best_monitor = float("inf")
        stale_epochs = 0
        if resumed_seed is not None:
            model.load_state_dict(cast(dict[str, Any], resumed_seed["model_state_dict"]))
            optimizer.load_state_dict(cast(dict[str, Any], resumed_seed["optimizer_state_dict"]))
            scheduler.load_state_dict(cast(dict[str, Any], resumed_seed["scheduler_state_dict"]))
            losses = list(cast(list[float], resumed_seed["losses"]))
            epoch_metrics = list(cast(list[dict[str, object]], resumed_seed["epochs"]))
            best_model_state = cast(dict[str, Any] | None, resumed_seed["best_model_state"])
            best_epoch = cast(int, resumed_seed["best_epoch"])
            best_monitor = cast(float, resumed_seed["best_monitor"])
            stale_epochs = cast(int, resumed_seed["stale_epochs"])
        graph_batches = _group_samples_by_graph(samples)
        for _ in range(len(losses), config.epochs):
            epoch_loss = 0.0
            epoch_classification_loss = 0.0
            epoch_return_loss = 0.0
            for batch in graph_batches:
                model.train()
                optimizer.zero_grad()
                outputs = model(batch[0].graph.data)
                loss, classification_loss, return_loss = _batch_loss(
                    outputs,
                    batch,
                    config,
                    classification_criterion,
                    huber,
                    torch,
                    return_transform_state,
                )
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise ValueError("DTSeatHGT training produced non-finite loss")
                loss.backward()
                grad_norm = None
                if config.gradient_clip_norm is not None:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                    if not bool(torch.isfinite(grad_norm_tensor).detach().cpu()):
                        raise ValueError("DTSeatHGT training produced non-finite gradient")
                    grad_norm = float(grad_norm_tensor.detach().cpu())
                else:
                    _require_finite_gradients(model, torch)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(batch)
                epoch_classification_loss += float(classification_loss.detach().cpu()) * len(batch)
                epoch_return_loss += float(return_loss.detach().cpu()) * len(batch)
            losses.append(epoch_loss / len(samples))
            validation_loss = _validation_loss(
                model,
                validation_samples,
                config,
                classification_criterion,
                huber,
                torch,
                return_transform_state,
            ) if validation_samples else losses[-1]
            if validation_samples:
                scheduler.step(validation_loss)
            if validation_loss < best_monitor:
                best_monitor = validation_loss
                best_epoch = len(losses)
                stale_epochs = 0
                best_model_state = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
            epoch_metrics.append(
                {
                    "epoch": len(losses),
                    "classification_loss": epoch_classification_loss / len(samples),
                    "return_loss": epoch_return_loss / len(samples),
                    "total_loss": losses[-1],
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": grad_norm,
                    "validation_loss": validation_loss if validation_samples else None,
                    **_epoch_observability(
                        model, samples, config.model.horizons, torch, return_transform_state
                    ),
                }
            )
            _save_training_progress(
                torch,
                progress_path,
                effective_run_id,
                samples,
                config,
                hashes,
                seed_runs,
                state_by_seed,
                {
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "losses": losses,
                    "epochs": epoch_metrics,
                    "best_model_state": best_model_state,
                    "best_epoch": best_epoch,
                    "best_monitor": best_monitor,
                    "stale_epochs": stale_epochs,
                },
            )
            if validation_samples and stale_epochs >= config.early_stopping_patience:
                break

        if best_model_state is None:
            raise ValueError("DTSeatHGT training did not produce a best model state")
        model.load_state_dict(best_model_state)

        ablation = _edge_weight_ablation(model, samples[0], config.edge_ablation_scale, torch)
        prediction_snapshot = _prediction_snapshot(
            model, samples, config.model.horizons, torch, return_transform_state
        )
        seed_runs.append(
            {
                "seed": seed,
                "losses": losses,
                "epochs": epoch_metrics,
                "best_epoch": best_epoch,
                "early_stopped": len(losses) < config.epochs,
                "graph_batch_count": len(graph_batches),
                "gradient_clip_norm": config.gradient_clip_norm,
                "last_gradient_norm": grad_norm,
                "edge_weight_ablation": ablation,
                "prediction_snapshot": prediction_snapshot,
            }
        )
        state_by_seed[str(seed)] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "runtime_data_scope": _runtime_data_scope(samples),
            "hashes": {
                **hashes,
                "model_state": _torch_state_hash(torch, model.state_dict()),
                "optimizer_state": _torch_state_hash(torch, optimizer.state_dict()),
            },
            "prediction_snapshot": prediction_snapshot,
        }
        _save_training_progress(
            torch,
            progress_path,
            effective_run_id,
            samples,
            config,
            hashes,
            seed_runs,
            state_by_seed,
            None,
        )

    if not state_by_seed:
        raise ValueError("no seed runs were executed")
    checkpoint = {
        "run_id": effective_run_id,
        "model_valid": False,
        "status": "completed",
        "config": config.to_dict(),
        "effective_training_config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "gradient_clip_norm": config.gradient_clip_norm,
            "loss_direction_weight": config.loss_direction_weight,
            "loss_return_weight": config.loss_return_weight,
            "return_loss": "huber",
            "edge_ablation_scale": config.edge_ablation_scale,
            "optimizer": "AdamW",
            "weight_decay": config.weight_decay,
            "scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": config.early_stopping_patience,
            "seeds": list(config.seeds),
        },
        "loss_history": {
            str(seed_run["seed"]): list(cast(Sequence[float], seed_run["losses"]))
            for seed_run in seed_runs
        },
        "feature_schema": samples[0].graph.metadata.get("feature_schema", {}),
        "feature_transform_state": (
            feature_transform_state.to_dict() if feature_transform_state is not None else None
        ),
        "feature_transform_diagnostics": feature_transform_diagnostics,
        "return_transform_state": (
            return_transform_state.to_dict() if return_transform_state is not None else None
        ),
        "return_transform_diagnostics": return_transform_diagnostics,
        "sample_ids": [sample.sample_id for sample in samples],
        "hashes": hashes,
        "schema": samples[0].graph.metadata,
        "label_summary": _label_summary(samples),
        "seed_runs": seed_runs,
        "state_by_seed": state_by_seed,
        "resume": {
            "restore_model_state_dict": True,
            "restore_optimizer_state_dict": True,
            "available_seeds": sorted(state_by_seed),
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(torch, checkpoint_path, checkpoint)
    if progress_path.exists():
        progress_path.unlink()
    return DTSeatHGTTrainingResult(
        run_id=effective_run_id,
        status="completed",
        model_valid=False,
        checkpoint_path=checkpoint_path,
        hashes={**hashes, "checkpoint": _hash_file(checkpoint_path)},
        metrics={
            "seed_count": len(config.seeds),
            "sample_count": len(samples),
            "graph_batch_count": len(_group_samples_by_graph(samples)),
            "seed_runs": seed_runs,
            "model_valid": False,
        },
    )


def restore_dt_seat_hgt_seed(
    checkpoint_path: str | Path,
    sample: TensorizedWindowSample,
    seed: int,
    *,
    feature_transform_state: FeatureTransformEvidence | None = None,
    return_transform_state: ReturnTransformState | None = None,
) -> RestoredDTSeatHGT:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise PyGModelUnavailableError(
            "DTSeatHGT restore requires PyTorch and torch-geometric. "
            "Install the ml extra before restoring."
        ) from error

    checkpoint = torch.load(Path(checkpoint_path), weights_only=False)
    if bool(checkpoint.get("model_valid")):
        raise ValueError("DTSeatHGT training checkpoint must not claim model_valid")
    expected_transform_hash = (
        feature_transform_state.fingerprint() if feature_transform_state is not None else None
    )
    actual_hashes = checkpoint.get("hashes")
    if not isinstance(actual_hashes, dict):
        raise ValueError("checkpoint is missing provenance hashes")
    if feature_transform_state is not None:
        if actual_hashes.get("feature_transform") != expected_transform_hash:
            raise ValueError("checkpoint feature transform does not match requested restore")
        if checkpoint.get("feature_transform_state") != feature_transform_state.to_dict():
            raise ValueError("checkpoint feature transform state does not match requested restore")
    if return_transform_state is not None:
        return_transform_state.validate()
        if actual_hashes.get("return_transform") != return_transform_state.fingerprint():
            raise ValueError("checkpoint return transform does not match requested restore")
        if checkpoint.get("return_transform_state") != return_transform_state.to_dict():
            raise ValueError("checkpoint return transform state does not match requested restore")
    config = _training_config_from_checkpoint(checkpoint)
    _validate_samples((sample,), config.model)
    seed_key = str(seed)
    state_by_seed = checkpoint.get("state_by_seed")
    if not isinstance(state_by_seed, dict) or seed_key not in state_by_seed:
        raise ValueError(f"checkpoint missing state for seed {seed}")
    seed_state = state_by_seed[seed_key]
    if not isinstance(seed_state, dict):
        raise ValueError(f"checkpoint state for seed {seed} is invalid")
    model = build_dt_seat_hgt(config.model, sample.graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.load_state_dict(seed_state["model_state_dict"])
    optimizer.load_state_dict(seed_state["optimizer_state_dict"])
    return RestoredDTSeatHGT(
        seed=seed,
        model=model,
        optimizer=optimizer,
        checkpoint=checkpoint,
    )


def prediction_snapshot_for_samples(
    model: Any,
    samples: Sequence[TensorizedWindowSample],
    horizons: Sequence[int],
    *,
    return_transform_state: ReturnTransformState | None = None,
) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise PyGModelUnavailableError(
            "DTSeatHGT prediction snapshot requires PyTorch and torch-geometric. "
            "Install the ml extra before restoring."
        ) from error
    return _prediction_snapshot(model, samples, horizons, torch, return_transform_state)


def _batch_loss(
    outputs: dict[str, object],
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    classification_criterion: Any,
    huber: Any,
    torch: Any,
    return_transform_state: ReturnTransformState | None,
) -> Any:
    if not samples:
        raise ValueError("DTSeatHGT graph batch must not be empty")
    direction_logits = outputs["direction_logits"]
    return_predictions = outputs["return_predictions"]
    if not isinstance(direction_logits, dict) or not isinstance(return_predictions, dict):
        raise ValueError("DTSeatHGT forward must return direction and return prediction dicts")
    target_index = torch.tensor(
        [sample.target_node_index for sample in samples], dtype=torch.long
    )
    classification_losses: list[Any] = []
    return_losses: list[Any] = []
    for horizon in config.model.horizons:
        if any(horizon not in sample.labels or horizon not in sample.returns for sample in samples):
            raise ValueError(f"DTSeatHGT graph batch is missing labels for horizon {horizon}")
        logits = direction_logits[horizon].index_select(0, target_index)
        predicted_returns = return_predictions[horizon].index_select(0, target_index)
        if config.model.direction_classes == 2:
            direction_target = torch.tensor(
                [float(sample.labels[horizon]) for sample in samples], dtype=logits.dtype
            )
        else:
            direction_target = torch.tensor(
                [int(sample.labels[horizon]) + 1 for sample in samples], dtype=torch.long
            )
        target_returns = [float(sample.returns[horizon]) for sample in samples]
        if return_transform_state is not None:
            target_returns = [
                return_transform_state.transform(horizon, value) for value in target_returns
            ]
        return_target = torch.tensor(
            target_returns,
            dtype=predicted_returns.dtype,
        )
        classification_losses.append(
            config.loss_direction_weight * classification_criterion(logits, direction_target)
        )
        return_losses.append(config.loss_return_weight * huber(predicted_returns, return_target))
    classification_loss = sum(classification_losses)
    return_loss = sum(return_losses)
    return classification_loss + return_loss, classification_loss, return_loss


def _validation_loss(
    model: Any,
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    classification_criterion: Any,
    huber: Any,
    torch: Any,
    return_transform_state: ReturnTransformState | None,
) -> float:
    if not samples:
        raise ValueError("validation samples must not be empty")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in _group_samples_by_graph(samples):
            loss, _, _ = _batch_loss(
                model(batch[0].graph.data),
                batch,
                config,
                classification_criterion,
                huber,
                torch,
                return_transform_state,
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise ValueError("DTSeatHGT validation produced non-finite loss")
            total += float(loss.detach().cpu()) * len(batch)
    return total / len(samples)


def _edge_weight_ablation(
    model: Any,
    sample: TensorizedWindowSample,
    scale: float,
    torch: Any,
) -> dict[str, object]:
    data = sample.graph.data
    originals = {
        edge_type: data[edge_type].edge_weight.clone()
        for edge_type in data.edge_types
        if hasattr(data[edge_type], "edge_weight")
    }
    model.eval()
    with torch.no_grad():
        baseline = model(data)
        for edge_type, edge_weight in originals.items():
            data[edge_type].edge_weight = edge_weight * scale
        perturbed = model(data)
        for edge_type, edge_weight in originals.items():
            data[edge_type].edge_weight = edge_weight
    deltas: dict[str, float] = {}
    for horizon in sample.labels:
        base = baseline["direction_logits"][horizon][sample.target_node_index]
        changed = perturbed["direction_logits"][horizon][sample.target_node_index]
        deltas[str(horizon)] = float(torch.abs(base - changed).max().detach().cpu())
    return {
        "scale": scale,
        "direction_logit_abs_delta_by_horizon": deltas,
        "edge_weights_consumed": any(value > 0 for value in deltas.values()),
    }


def _prediction_snapshot(
    model: Any,
    samples: Sequence[TensorizedWindowSample],
    horizons: Sequence[int],
    torch: Any,
    return_transform_state: ReturnTransformState | None = None,
) -> dict[str, object]:
    model.eval()
    snapshot: dict[str, object] = {}
    with torch.no_grad():
        for batch in _group_samples_by_graph(samples):
            outputs = model(batch[0].graph.data)
            for sample in batch:
                snapshot[sample.sample_id] = {
                    "direction_logits": {
                        str(horizon): _serialize_direction_logits(
                            outputs["direction_logits"][horizon][sample.target_node_index]
                        )
                        for horizon in horizons
                    },
                    "return_predictions": _return_predictions_for_sample(
                        outputs, sample, horizons, torch, return_transform_state
                    ),
                }
    return snapshot


def _return_predictions_for_sample(
    outputs: dict[str, object],
    sample: TensorizedWindowSample,
    horizons: Sequence[int],
    torch: Any,
    return_transform_state: ReturnTransformState | None,
) -> dict[str, float]:
    raw_predictions = outputs["return_predictions"]
    if not isinstance(raw_predictions, dict):
        raise ValueError("DTSeatHGT forward must return return predictions")
    result: dict[str, float] = {}
    for horizon in horizons:
        value = float(raw_predictions[horizon][sample.target_node_index].detach().cpu())
        result[str(horizon)] = (
            return_transform_state.inverse_transform(horizon, value)
            if return_transform_state is not None
            else value
        )
    return result


def _epoch_observability(
    model: Any,
    samples: Sequence[TensorizedWindowSample],
    horizons: Sequence[int],
    torch: Any,
    return_transform_state: ReturnTransformState | None,
) -> dict[str, object]:
    snapshot = _prediction_snapshot(model, samples, horizons, torch, return_transform_state)
    logits = _flatten_direction_logits(
        _snapshot_value(snapshot, sample.sample_id, "direction_logits", horizon)
        for sample in samples
        for horizon in horizons
    )
    returns = [
        _float_snapshot_value(
            _snapshot_value(snapshot, sample.sample_id, "return_predictions", horizon)
        )
        for sample in samples
        for horizon in horizons
    ]
    labels = [float(sample.returns[horizon]) for sample in samples for horizon in horizons]
    probabilities = [_sigmoid(value) for value in logits]
    return {
        "logits": _distribution(logits),
        "probabilities": {
            **_distribution(probabilities),
            "saturation_ratio": sum(value <= 0.01 or value >= 0.99 for value in probabilities)
            / len(probabilities),
        },
        "predicted_returns": _distribution(returns),
        "label_return_std": _distribution(labels)["std"],
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("observability values must not be empty")
    mean = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
    }


def _serialize_direction_logits(value: Any) -> float | list[float]:
    flattened = value.detach().cpu().reshape(-1).tolist()
    if len(flattened) == 1:
        return float(flattened[0])
    return [float(item) for item in flattened]


def _flatten_direction_logits(values: Iterable[float | list[float]]) -> list[float]:
    flattened: list[float] = []
    for value in values:
        flattened.extend(value if isinstance(value, list) else [value])
    return flattened


def _snapshot_value(
    snapshot: dict[str, object], sample_id: str, head: str, horizon: int
) -> float | list[float]:
    raw_sample = snapshot.get(sample_id)
    if not isinstance(raw_sample, dict):
        raise ValueError("observability snapshot is missing a sample")
    raw_head = raw_sample.get(head)
    if not isinstance(raw_head, dict):
        raise ValueError("observability snapshot is missing a model head")
    value = raw_head.get(str(horizon))
    if not isinstance(value, (float, list)) or (
        isinstance(value, list) and not all(isinstance(item, float) for item in value)
    ):
        raise ValueError("observability snapshot is missing a horizon value")
    return value


def _float_snapshot_value(value: float | list[float]) -> float:
    if isinstance(value, list):
        raise ValueError("expected scalar model head value")
    return value


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value)) if value >= 0 else exp(value) / (1.0 + exp(value))


def _group_samples_by_graph(
    samples: Sequence[TensorizedWindowSample],
) -> tuple[tuple[TensorizedWindowSample, ...], ...]:
    grouped: dict[int, list[TensorizedWindowSample]] = {}
    for sample in samples:
        grouped.setdefault(id(sample.graph), []).append(sample)
    return tuple(tuple(batch) for batch in grouped.values())


def _validate_samples(
    samples: Sequence[TensorizedWindowSample], config: DTSeatHGTConfig
) -> None:
    expected_metadata = samples[0].graph.data.metadata()
    for sample in samples:
        expected_schema = (
            LEGACY_BINARY_LABEL_SCHEMA
            if config.direction_classes == 2
            else EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA
        )
        if sample.label_schema_version != expected_schema:
            raise ValueError(
                "DTSeatHGT direction head and tensorized label schema do not match"
            )
        data = sample.graph.data
        if config.target_node_type not in data.node_types:
            raise ValueError(f"sample {sample.sample_id} missing target node type")
        target_count = int(data[config.target_node_type].x.size(0))
        if sample.target_node_index < 0 or sample.target_node_index >= target_count:
            raise ValueError(f"sample {sample.sample_id} target_node_index is out of range")
        if data.metadata() != expected_metadata:
            raise ValueError("all DTSeatHGT training samples must share the same graph schema")
        if not data.edge_types:
            raise ValueError("DTSeatHGT training requires at least one edge type")
        for edge_type in data.edge_types:
            if not hasattr(data[edge_type], "edge_weight"):
                raise ValueError(f"sample {sample.sample_id} missing edge_weight for {edge_type}")
        for horizon in config.horizons:
            if horizon not in sample.labels or horizon not in sample.returns:
                raise ValueError(f"sample {sample.sample_id} missing labels for horizon {horizon}")


def _require_non_empty_samples(samples: Sequence[TensorizedWindowSample]) -> None:
    if not samples:
        raise ValueError("DTSeatHGT training requires at least one tensorized sample")


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _hashes(
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    validation_samples: Sequence[TensorizedWindowSample] = (),
) -> dict[str, str]:
    hashes = {
        "config": _hash_json(config.to_dict()),
        "sample": _hash_json([_sample_manifest(sample) for sample in samples]),
        "label": _hash_json([_label_payload(sample) for sample in samples]),
        "schema": _hash_json(samples[0].graph.metadata),
    }
    if validation_samples:
        hashes["validation_sample"] = _hash_json(
            [_sample_manifest(sample) for sample in validation_samples]
        )
        hashes["validation_label"] = _hash_json(
            [_label_payload(sample) for sample in validation_samples]
        )
    return hashes


def _sample_manifest(sample: TensorizedWindowSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "source_hash": sample.source_hash,
        "target_stock_id": sample.target_stock_id,
        "target_node_index": sample.target_node_index,
        "label_schema_version": sample.label_schema_version,
        "graph_metadata": sample.graph.metadata,
    }


def _label_payload(sample: TensorizedWindowSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "labels": dict(sorted(sample.labels.items())),
        "returns": dict(sorted(sample.returns.items())),
    }


def _label_summary(samples: Sequence[TensorizedWindowSample]) -> dict[str, object]:
    return {
        sample.sample_id: {
            "source_hash": sample.source_hash,
            "target_stock_id": sample.target_stock_id,
            "labels": dict(sorted(sample.labels.items())),
            "returns": dict(sorted(sample.returns.items())),
        }
        for sample in samples
    }


def _runtime_data_scope(samples: Sequence[TensorizedWindowSample]) -> dict[str, object]:
    return {
        "scope": "sample_only",
        "sample_source_hashes": {
            sample.sample_id: sample.source_hash for sample in samples
        },
    }


def _training_config_from_checkpoint(
    checkpoint: dict[str, object],
) -> DTSeatHGTTrainingConfig:
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint missing training config")
    raw_model = raw.get("model")
    if not isinstance(raw_model, dict):
        raise ValueError("checkpoint missing model config")
    model_config = DTSeatHGTConfig(
        hidden_dim=int(raw_model["hidden_dim"]),
        num_heads=int(raw_model["num_heads"]),
        num_layers=int(raw_model["num_layers"]),
        dropout=float(raw_model["dropout"]),
        horizons=tuple(int(item) for item in raw_model["horizons"]),
        target_node_type=str(raw_model.get("target_node_type", "stock")),
        direction_classes=int(raw_model.get("direction_classes", 2)),
    )
    return DTSeatHGTTrainingConfig(
        model=model_config,
        learning_rate=float(raw["learning_rate"]),
        epochs=int(raw["epochs"]),
        seeds=tuple(int(item) for item in raw["seeds"]),
        loss_direction_weight=float(raw["loss_direction_weight"]),
        loss_return_weight=float(raw["loss_return_weight"]),
        edge_ablation_scale=float(raw["edge_ablation_scale"]),
        gradient_clip_norm=(
            None
            if raw.get("gradient_clip_norm") is None
            else float(raw["gradient_clip_norm"])
        ),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 12)),
        scheduler_patience=int(raw.get("scheduler_patience", 4)),
    )


def _require_finite_gradients(model: Any, torch: Any) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all().detach().cpu()
        ):
            raise ValueError("DTSeatHGT training produced non-finite gradient")


def _verified_existing_result(
    torch: Any,
    checkpoint_path: Path,
    run_id: str,
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    expected_hashes: dict[str, str],
) -> DTSeatHGTTrainingResult:
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except Exception as error:
        raise ValueError("existing DTSeatHGT checkpoint is unreadable") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("existing DTSeatHGT checkpoint must contain an object")
    if checkpoint.get("run_id") != run_id or checkpoint.get("status") != "completed":
        raise ValueError("existing DTSeatHGT checkpoint identity or status mismatch")
    if checkpoint.get("model_valid") is not False:
        raise ValueError("existing DTSeatHGT checkpoint must remain model_valid=false")
    if checkpoint.get("hashes") != expected_hashes:
        raise ValueError("existing DTSeatHGT checkpoint hashes do not match requested run")
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("existing DTSeatHGT checkpoint config does not match requested run")
    if checkpoint.get("sample_ids") != [sample.sample_id for sample in samples]:
        raise ValueError("existing DTSeatHGT checkpoint samples do not match requested run")
    state_by_seed = checkpoint.get("state_by_seed")
    if not isinstance(state_by_seed, dict) or set(state_by_seed) != {
        str(seed) for seed in config.seeds
    }:
        raise ValueError("existing DTSeatHGT checkpoint seed states are incomplete")
    seed_runs = checkpoint.get("seed_runs")
    if not isinstance(seed_runs, list):
        raise ValueError("existing DTSeatHGT checkpoint seed metrics are missing")
    return DTSeatHGTTrainingResult(
        run_id=run_id,
        status="completed",
        model_valid=False,
        checkpoint_path=checkpoint_path,
        hashes={**expected_hashes, "checkpoint": _hash_file(checkpoint_path)},
        metrics={
            "seed_count": len(config.seeds),
            "sample_count": len(samples),
            "graph_batch_count": len(_group_samples_by_graph(samples)),
            "seed_runs": seed_runs,
            "model_valid": False,
            "resumed": True,
        },
    )


def _load_training_progress(
    torch: Any,
    progress_path: Path,
    run_id: str,
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    expected_hashes: dict[str, str],
) -> dict[str, object]:
    try:
        progress = torch.load(progress_path, weights_only=False)
    except Exception as error:
        raise ValueError("in-progress DTSeatHGT checkpoint is unreadable") from error
    if not isinstance(progress, dict):
        raise ValueError("in-progress DTSeatHGT checkpoint must contain an object")
    if progress.get("status") != "in_progress" or progress.get("run_id") != run_id:
        raise ValueError("in-progress DTSeatHGT checkpoint identity or status mismatch")
    if progress.get("model_valid") is not False:
        raise ValueError("in-progress DTSeatHGT checkpoint must remain model_valid=false")
    if progress.get("hashes") != expected_hashes:
        raise ValueError("in-progress DTSeatHGT checkpoint hashes do not match requested run")
    if progress.get("config") != config.to_dict():
        raise ValueError("in-progress DTSeatHGT checkpoint config does not match requested run")
    if progress.get("sample_ids") != [sample.sample_id for sample in samples]:
        raise ValueError("in-progress DTSeatHGT checkpoint samples do not match requested run")
    if not isinstance(progress.get("seed_runs"), list) or not isinstance(
        progress.get("state_by_seed"), dict
    ):
        raise ValueError("in-progress DTSeatHGT checkpoint seed state is malformed")
    return cast(dict[str, object], progress)


def _progress_for_seed(progress: dict[str, object] | None, seed: int) -> dict[str, object] | None:
    if progress is None:
        return None
    active = progress.get("active_seed")
    if not isinstance(active, dict) or active.get("seed") != seed:
        return None
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "losses",
        "epochs",
        "best_model_state",
        "best_epoch",
        "best_monitor",
        "stale_epochs",
    }
    if not required.issubset(active):
        raise ValueError("in-progress DTSeatHGT checkpoint active seed is incomplete")
    return cast(dict[str, object], active)


def _save_training_progress(
    torch: Any,
    progress_path: Path,
    run_id: str,
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    hashes: dict[str, str],
    seed_runs: Sequence[dict[str, object]],
    state_by_seed: Mapping[str, dict[str, object]],
    active_seed: dict[str, object] | None,
) -> None:
    _atomic_torch_save(
        torch,
        progress_path,
        {
            "run_id": run_id,
            "model_valid": False,
            "status": "in_progress",
            "config": config.to_dict(),
            "sample_ids": [sample.sample_id for sample in samples],
            "hashes": hashes,
            "seed_runs": list(seed_runs),
            "state_by_seed": dict(state_by_seed),
            "active_seed": active_seed,
        },
    )


def _hash_json(payload: object) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _torch_state_hash(torch: Any, state: object) -> str:
    stream = BytesIO()
    torch.save(state, stream)
    return sha256(stream.getvalue()).hexdigest()


def _atomic_torch_save(torch: Any, path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
