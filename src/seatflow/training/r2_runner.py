"""Serial, one-snapshot-at-a-time GPU runner for SeatRankLite."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seatflow.graph.tensorize import TensorizedGraph
from seatflow.models.seat_rank_lite import (
    SeatRankLiteConfig,
    build_seat_rank_lite,
    validate_parameter_count,
)
from seatflow.runtime.device import (
    move_graph_to_device,
    release_cuda_objects,
    report_cuda_memory,
    resolve_device,
)
from seatflow.training.amp import build_amp_controller
from seatflow.training.ranking_losses import compute_ranking_loss
from seatflow.training.seed_lifecycle import save_seed_artifacts


@dataclass(frozen=True)
class R2GraphBatch:
    graph: TensorizedGraph
    residual_targets: Any
    direction_targets: Any
    prediction_date: str


@dataclass(frozen=True)
class R2TrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 4
    mixed_precision: bool = True
    preferred_dtype: str = "fp16"
    device: str = "auto"
    hard_memory_bytes: int = int(7.4 * 1024**3)
    max_pairs_per_date: int = 1024

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("epochs and gradient accumulation must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning rate and gradient clip norm must be positive")
        if self.early_stopping_patience <= 0 or self.hard_memory_bytes <= 0:
            raise ValueError("patience and hard memory limit must be positive")


@dataclass(frozen=True)
class R2SeedResult:
    fold_id: int
    seed: int
    device: str
    checkpoint_path: Path
    parameter_count: int
    epochs_completed: int
    step_metrics: tuple[dict[str, object], ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocated_after_release_bytes: int


def train_seat_rank_lite_seed(
    batches: tuple[R2GraphBatch, ...],
    model_config: SeatRankLiteConfig,
    training_config: R2TrainingConfig,
    output_root: str | Path,
    *,
    fold_id: int,
    seed: int,
    validation_batches: tuple[R2GraphBatch, ...] = (),
) -> R2SeedResult:
    if not batches:
        raise ValueError("R2 training requires at least one graph batch")
    import torch

    _set_seed(seed, torch)
    device = resolve_device(training_config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_seat_rank_lite(model_config, batches[0].graph).to(device)
    parameter_report = validate_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        foreach=False,
    )
    amp = build_amp_controller(
        device,
        mixed_precision=training_config.mixed_precision,
        preferred_dtype=training_config.preferred_dtype,
    )
    step_metrics: list[dict[str, object]] = []
    best_monitor = float("-inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    epochs_completed = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(training_config.epochs):
        model.train()
        for batch_index, batch in enumerate(batches):
            data = move_graph_to_device(batch.graph.data.clone(), device)
            residual_targets = batch.residual_targets.to(device)
            direction_targets = batch.direction_targets.to(device)
            with amp.autocast():
                outputs = model(data)
                _require_nonconstant_predictions(outputs["rank_score"])
                loss, components = compute_ranking_loss(
                    outputs,
                    residual_targets,
                    direction_targets,
                    (batch.prediction_date,) * int(residual_targets.numel()),
                    max_pairs_per_date=training_config.max_pairs_per_date,
                )
                scaled_loss = loss / training_config.gradient_accumulation_steps
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise ValueError("R2 training produced non-finite loss")
            amp.backward(scaled_loss)
            should_step = (
                (batch_index + 1) % training_config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(batches)
            )
            gradient_norm: float | None = None
            if should_step:
                amp.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip_norm
                )
                if not bool(torch.isfinite(norm).detach().cpu()):
                    raise ValueError("R2 training produced non-finite gradients")
                gradient_norm = float(norm.detach().cpu())
                amp.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
            memory = report_cuda_memory(device)
            if max(memory.peak_allocated_bytes, memory.peak_reserved_bytes) > (
                training_config.hard_memory_bytes
            ):
                raise MemoryError("R2 CUDA memory hard limit exceeded")
            step_metrics.append(
                {
                    "epoch": epoch + 1,
                    "prediction_date": batch.prediction_date,
                    "parameter_count": parameter_report["parameter_count"],
                    "node_count": sum(
                        int(data[node_type].num_nodes) for node_type in data.node_types
                    ),
                    "edge_count": sum(
                        int(data[edge_type].edge_index.size(1)) for edge_type in data.edge_types
                    ),
                    "loss": float(loss.detach().cpu()),
                    "component_losses": {
                        name: float(value.detach().cpu()) for name, value in components.items()
                    },
                    "gradient_norm": gradient_norm,
                    "peak_allocated_bytes": memory.peak_allocated_bytes,
                    "peak_reserved_bytes": memory.peak_reserved_bytes,
                    "finite": True,
                }
            )
            del data, residual_targets, direction_targets, outputs, loss, scaled_loss, components
            release_cuda_objects(device=device)
        epochs_completed = epoch + 1
        monitor = (
            _validation_monitor(model, validation_batches, device, amp)
            if validation_batches
            else -sum(
                float(row["loss"])
                for row in step_metrics
                if row["epoch"] == epoch + 1
            )
        )
        if monitor > best_monitor:
            best_monitor = monitor
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if validation_batches and stale_epochs >= training_config.early_stopping_patience:
            break
    if best_state is None:
        raise ValueError("R2 training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    predictions = _prediction_snapshot(model, batches, device, amp)
    checkpoint_path = save_seed_artifacts(
        output_root,
        fold_id=fold_id,
        seed=seed,
        model=model,
        predictions=predictions,
        metadata={
            "device": str(device),
            "epochs_completed": epochs_completed,
            "parameter_report": parameter_report,
            "step_metrics": step_metrics,
            "model_valid": False,
        },
    )
    peak = report_cuda_memory(device)
    parameter_count = int(parameter_report["parameter_count"])
    del model, optimizer, amp, best_state
    after_release = release_cuda_objects(device=device)
    return R2SeedResult(
        fold_id=fold_id,
        seed=seed,
        device=str(device),
        checkpoint_path=checkpoint_path,
        parameter_count=parameter_count,
        epochs_completed=epochs_completed,
        step_metrics=tuple(step_metrics),
        peak_allocated_bytes=peak.peak_allocated_bytes,
        peak_reserved_bytes=peak.peak_reserved_bytes,
        allocated_after_release_bytes=after_release.allocated_bytes,
    )


def _prediction_snapshot(
    model: Any,
    batches: tuple[R2GraphBatch, ...],
    device: Any,
    amp: Any,
) -> dict[str, object]:
    import torch

    model.eval()
    result: dict[str, object] = {}
    with torch.no_grad():
        for batch in batches:
            data = move_graph_to_device(batch.graph.data.clone(), device)
            with amp.autocast():
                outputs = model(data)
            result[batch.prediction_date] = {
                name: outputs[name].detach().float().cpu().tolist()
                for name in (
                    "rank_score",
                    "residual_return",
                    "direction_logit",
                    "uncertainty",
                )
            }
            del data, outputs
            release_cuda_objects(device=device)
    return result


def _validation_monitor(
    model: Any,
    batches: tuple[R2GraphBatch, ...],
    device: Any,
    amp: Any,
) -> float:
    import torch

    model.eval()
    daily_scores: list[float] = []
    with torch.no_grad():
        for batch in batches:
            data = move_graph_to_device(batch.graph.data.clone(), device)
            targets = batch.residual_targets.to(device)
            with amp.autocast():
                predictions = model(data)["rank_score"].float()
            predicted_ranks = predictions.argsort().argsort().float()
            target_ranks = targets.argsort().argsort().float()
            rank_ic = _correlation(predicted_ranks, target_ranks)
            topk = min(5, int(predictions.numel()))
            selected = targets[predictions.topk(topk).indices].mean()
            standardized_spread = (selected - targets.mean()) / targets.std().clamp_min(1e-8)
            daily_scores.append(float((0.6 * rank_ic + 0.4 * standardized_spread).cpu()))
            del data, targets, predictions
            release_cuda_objects(device=device)
    return sum(daily_scores) / len(daily_scores)


def _correlation(left: Any, right: Any) -> Any:
    left = left - left.mean()
    right = right - right.mean()
    return (left * right).sum() / (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1e-8)


def _require_nonconstant_predictions(predictions: Any) -> None:
    import torch

    if not bool(torch.isfinite(predictions).all().detach().cpu()):
        raise ValueError("R2 model produced non-finite predictions")
    if predictions.numel() > 1 and float(predictions.detach().float().std().cpu()) <= 1e-10:
        raise ValueError("R2 model produced constant rank predictions")


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
