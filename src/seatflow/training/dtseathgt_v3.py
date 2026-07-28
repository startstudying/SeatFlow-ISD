"""Three-stage training for DT-SeatHGT v3.

Stage A: Train base branches only (freeze seat)
Stage B: Train seat residual branch (freeze base, use detached residual target)
Stage C: Joint fine-tune all (unfreeze all at 0.1x LR)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from pathlib import Path
from typing import Any, cast

import torch

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.models.dt_seat_hgt_v3 import DTSeatHGTv3Config, DTSeatHGTv3Model, build_dt_seat_hgt_v3
from seatflow.models.seat_hgt import PyGModelUnavailableError
from seatflow.training.v4_loss import V4Loss, V4LossWeights

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Three-stage training phases for DT-SeatHGT v3."""

    BASE_ONLY = auto()  # Stage 1: Train base branches only
    SEAT_RESIDUAL = auto()  # Stage 2: Seat residual + seat_gate
    JOINT_FINETUNE = auto()  # Stage 3: Joint fine-tune all


@dataclass(frozen=True)
class DTSeatHGTv3TrainingStageConfig:
    """Configuration for a single training stage."""

    stage: TrainingStage
    epochs: int
    learning_rate: float
    freeze_base: bool = False
    freeze_seat: bool = False
    early_stopping_patience: int = 4


@dataclass(frozen=True)
class DTSeatHGTv3TrainingConfig:
    """Three-stage training configuration for DT-SeatHGT v3."""

    model: DTSeatHGTv3Config
    stage1: DTSeatHGTv3TrainingStageConfig
    stage2: DTSeatHGTv3TrainingStageConfig
    stage3: DTSeatHGTv3TrainingStageConfig
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    seeds: tuple[int, ...] = (2026,)
    loss_direction_weight: float = 1.0
    loss_return_weight: float = 1.0
    direction_classes: int = 2  # Binary (2) or multi-class (>2)
    use_amp: bool = True  # Automatic mixed precision
    accumulation_steps: int = 1  # Gradient accumulation
    validation_split: float = 0.15  # Fraction of samples for validation
    early_stopping_patience: int = 4  # Epochs to wait before early stopping

    def __post_init__(self) -> None:
        if self.loss_direction_weight < 0 or self.loss_return_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.loss_direction_weight == 0 and self.loss_return_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        for stage_config in (self.stage1, self.stage2, self.stage3):
            if stage_config.epochs <= 0:
                raise ValueError(f"{stage_config.stage.name} epochs must be positive")
            if stage_config.learning_rate <= 0:
                raise ValueError(f"{stage_config.stage.name} learning_rate must be positive")


@dataclass(frozen=True)
class DTSeatHGTv3TrainingResult:
    """Result from three-stage training."""

    run_id: str
    status: str
    model_valid: bool
    checkpoint_path: Path
    stage_results: dict[str, dict[str, Any]]
    total_epochs: int
    final_loss: float
    seed: int
    gpu_memory_peak: float | None = None


def train_dt_seat_hgt_v3(
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTv3TrainingConfig,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    device: str | None = None,
    validation_samples: Sequence[TensorizedWindowSample] | None = None,
) -> list[DTSeatHGTv3TrainingResult]:
    """Train DT-SeatHGT v3 with three-stage curriculum.

    Stage A: Train base branches (stock GRU + static MLP + base fusion)
    Stage B: Train seat residual branch + seat_gate (freeze base)
    Stage C: Joint fine-tune all components

    Args:
        samples: Tensorized window samples for training
        config: Three-stage training configuration
        output_root: Output directory for checkpoints
        run_id: Run identifier
        device: Device to train on (auto-detect if None)

    Returns:
        Training result with stage metrics and checkpoint path
    """
    try:
        import torch
    except ModuleNotFoundError as error:
        raise PyGModelUnavailableError(
            "DT-SeatHGT v3 training requires PyTorch. Install the ml extra before training."
        ) from error

    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from uuid import uuid4

    # Results across seeds
    all_results: list[DTSeatHGTv3TrainingResult] = []

    for seed in config.seeds:
        effective_run_id = f"{run_id or 'dtseathgt-v3'}-{seed}-{uuid4().hex[:8]}"
        output_path = (
            Path(output_root) / f"seed_{seed}"
            if output_root
            else Path.cwd() / f"seed_{seed}"
        )
        output_path.mkdir(parents=True, exist_ok=True)

        # Set seed for this run (covers all RNG sources so different seeds
        # produce genuinely different initializations and data orderings).
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ModuleNotFoundError:
            pass

        # Initialize model ONCE before optimizer
        model = build_dt_seat_hgt_v3(
            config.model, cast("dict[str, Any] | None", samples[0].graph)
        )
        model = model.to(device)

        # Validate parameter count
        param_count = model.parameter_count()
        logger.info(f"Seed {seed}: Model has {param_count:,} parameters")

        # Split samples for validation if not provided
        if validation_samples is None and config.validation_split > 0:
            # Sort samples by as_of_date for time-series integrity
            sorted_samples = sorted(
                samples, key=lambda s: (s.as_of_date or date.min, s.sample_id)
            )
            split_idx = int(len(sorted_samples) * (1 - config.validation_split))

            # Ensure at least 1 sample in training set
            if split_idx == 0 and len(samples) > 0:
                split_idx = 1
            if split_idx == len(samples) and len(samples) > 1:
                split_idx = len(samples) - 1

            # Time-sorted split: earlier samples for train, latest for validation
            train_samples = sorted_samples[:split_idx]
            val_samples = sorted_samples[split_idx:]
            logger.info(
                f"Seed {seed}: Split {len(samples)} samples into "
                f"{len(train_samples)} train / {len(val_samples)} val "
                f"(time-sorted, no random shuffle)"
            )
        else:
            train_samples = list(samples)
            val_samples = list(validation_samples) if validation_samples else []

        # Track GPU memory (only when actually running on CUDA)
        gpu_memory_peak: float | None = None
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            gpu_memory_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GiB

        # Train each stage
        stage_results: dict[str, dict[str, Any]] = {}
        total_epochs = 0
        final_loss = 0.0

        for stage_config in (config.stage1, config.stage2, config.stage3):
            stage_name = stage_config.stage.name
            logger.info(f"Seed {seed}: Starting {stage_name}")

            stage_result = _train_single_stage(
                model=model,
                train_samples=train_samples,
                val_samples=val_samples if val_samples else None,
                stage_config=stage_config,
                stage_name=stage_name,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                loss_direction_weight=config.loss_direction_weight,
                loss_return_weight=config.loss_return_weight,
                direction_classes=config.direction_classes,
                use_amp=config.use_amp,
                accumulation_steps=config.accumulation_steps,
                device=device,
                early_stopping_patience=config.early_stopping_patience,
            )

            stage_results[stage_name] = stage_result
            total_epochs += stage_result["epochs"]
            final_loss = stage_result["loss"]

            # Track GPU memory (only when actually running on CUDA)
            if device == "cuda" and torch.cuda.is_available():
                current_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if gpu_memory_peak is not None:
                    gpu_memory_peak = max(gpu_memory_peak, current_peak)

            logger.info(f"Seed {seed}: Completed {stage_name} - Loss: {final_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_path / "dtseathgt_v3_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.model.to_dict(),
                "run_id": effective_run_id,
                "stage_results": stage_results,
                "seed": seed,
            },
            checkpoint_path,
        )

        # Store result
        result = DTSeatHGTv3TrainingResult(
            run_id=effective_run_id,
            status="completed",
            model_valid=False,  # R5: must remain False until final evaluation
            checkpoint_path=checkpoint_path,
            stage_results=stage_results,
            total_epochs=total_epochs,
            final_loss=final_loss,
            seed=seed,
            gpu_memory_peak=gpu_memory_peak,
        )
        all_results.append(result)

        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Return aggregated results from ALL seeds
    if not all_results:
        raise RuntimeError("No training results collected")
    return all_results


def _v4_loss_weights_for_stage(stage: TrainingStage) -> V4LossWeights:
    """Per-stage V4Loss weights.

    Stage A (BASE_ONLY): train base directly; seat branch frozen so its
        contribution is zero and seat_weight is ineffective. listwise on the
        composite (base+0) still trains the base for ranking.
    Stage B (SEAT_RESIDUAL): base frozen. seat_weight drives the residual
        branch; listwise on the composite trains the seat delta for ranking.
        base_weight is left at 1.0 but the base branch is frozen, so base_loss
        produces no gradient into base params (residual_seat_loss detaches
        base_pred internally regardless).
    Stage C (JOINT_FINETUNE): all branches trainable, full composite ranking.
    """
    if stage == TrainingStage.BASE_ONLY:
        return V4LossWeights(base_weight=1.0, seat_weight=0.0, listwise_weight=0.5)
    if stage == TrainingStage.SEAT_RESIDUAL:
        return V4LossWeights(base_weight=1.0, seat_weight=1.0, listwise_weight=0.5)
    return V4LossWeights(base_weight=1.0, seat_weight=1.0, listwise_weight=0.5)


def _group_samples_by_date(
    samples: Sequence[TensorizedWindowSample],
) -> list[list[int]]:
    """Group sample indices by as_of_date, preserving date order.

    Returns a list of buckets; each bucket is a list of indices into
    ``samples`` sharing the same as_of_date (one cross-section). Date order is
    preserved so training sees time-sorted cross-sections (no future leakage).
    """
    order: dict[date, list[int]] = {}
    for idx, sample in enumerate(samples):
        sample_date = cast(date, sample.as_of_date)
        order.setdefault(sample_date, []).append(idx)
    return [order[d] for d in sorted(order)]


def _compute_v4_loss_for_buckets(
    model: DTSeatHGTv3Model,
    samples: Sequence[TensorizedWindowSample],
    buckets: list[list[int]],
    v4_loss: V4Loss,
    *,
    enable_seat_branch: bool,
    direction_classes: int,
    direction_weight: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward every sample in each date bucket and compute the V4Loss.

    Buckets are processed one at a time; each bucket's per-sample predictions
    are gathered into 1-D tensors so listwise_rank_loss sees a real
    cross-section list. The returned loss is the mean bucket loss (preserving
    the autograd graph). Component means are returned for logging.

    Single-element buckets contribute zero listwise signal (handled inside
    listwise_rank_loss) but still contribute base/seat regression loss.
    """
    bucket_losses: list[torch.Tensor] = []
    component_sums: dict[str, float] = {}
    n_samples = 0

    for bucket in buckets:
        base_preds: list[torch.Tensor] = []
        seat_residual_preds: list[torch.Tensor] = []
        target_returns: list[torch.Tensor] = []
        direction_logits_list: list[torch.Tensor] = []
        direction_labels_list: list[torch.Tensor] = []
        dates: list[date] = []

        for sample_idx in bucket:
            sample = samples[sample_idx]
            graph_data = sample.graph.data
            if hasattr(graph_data, "to"):
                graph_data = graph_data.to(device)

            outputs = model(graph_data, enable_seat_branch=enable_seat_branch)

            target_idx: int | torch.Tensor = sample.target_node_index
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor([target_idx], dtype=torch.long, device=device)
            else:
                target_idx = target_idx.to(device)

            base_pred = outputs["base_return_prediction"][target_idx]
            # seat residual = gated seat contribution (gate detached to keep
            # seat residual semantics consistent with V4-A8 Stage B).
            seat_gate = outputs["seat_gate"][target_idx].detach()
            seat_delta = outputs["seat_return_delta"][target_idx]
            seat_residual_pred = seat_gate * seat_delta

            base_preds.append(base_pred)
            seat_residual_preds.append(seat_residual_pred)
            target_returns.append(
                torch.tensor(
                    [float(sample.returns.get(1, 0.0))],
                    dtype=torch.float32,
                    device=device,
                )
            )
            direction_logits_list.append(outputs["direction_logit"][target_idx])
            direction_labels_list.append(
                torch.tensor(float(sample.labels.get(1, 0)), dtype=torch.float32, device=device)
            )
            dates.append(cast(date, sample.as_of_date))

        base_pred_t = torch.stack(base_preds).reshape(-1)
        seat_residual_t = torch.stack(seat_residual_preds).reshape(-1)
        target_t = torch.stack(target_returns).reshape(-1)
        direction_logits_t = torch.stack(direction_logits_list).reshape(-1)
        direction_labels_t = torch.stack(direction_labels_list).reshape(-1)

        out = v4_loss(
            base_pred=base_pred_t,
            seat_residual_pred=seat_residual_t,
            target_return=target_t,
            dates=dates,
            direction_logits=direction_logits_t,
            direction_labels=direction_labels_t,
            direction_classes=direction_classes,
            seat_gate=None,
        )
        # Add direction loss separately so its weight is controlled by the
        # legacy loss_direction_weight knob (V4Loss.direction_weight defaults 0).
        if direction_weight > 0:
            from torch.nn import functional as F

            if direction_classes == 2:
                binary_labels = (direction_labels_t > 0).float()
                dir_loss = F.binary_cross_entropy_with_logits(
                    direction_logits_t, binary_labels
                )
            else:
                class_labels = direction_labels_t.long() + 1
                dir_loss = F.cross_entropy(
                    direction_logits_t.unsqueeze(-1), class_labels
                )
            out["direction_loss"] = dir_loss
            out["loss"] = out["loss"] + direction_weight * dir_loss

        bucket_losses.append(out["loss"])
        n_samples += len(bucket)
        for key, value in out.items():
            contribution = float(value.detach().cpu()) * len(bucket)
            component_sums[key] = component_sums.get(key, 0.0) + contribution

    if not bucket_losses:
        # Should not happen, but keep the graph connected for safety.
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {}

    total_loss = torch.stack(bucket_losses).mean()
    component_means = {k: v / max(n_samples, 1) for k, v in component_sums.items()}
    return total_loss, component_means


def _train_single_stage(
    model: DTSeatHGTv3Model,
    train_samples: Sequence[TensorizedWindowSample],
    stage_config: DTSeatHGTv3TrainingStageConfig,
    stage_name: str,
    weight_decay: float,
    gradient_clip_norm: float | None,
    loss_direction_weight: float,
    loss_return_weight: float,
    direction_classes: int,
    use_amp: bool,
    accumulation_steps: int,
    device: str,
    val_samples: Sequence[TensorizedWindowSample] | None = None,
    early_stopping_patience: int = 4,
) -> dict[str, Any]:
    """Train for a single stage with appropriate freeze masks.

    Args:
        model: The model to train
        train_samples: Training samples
        val_samples: Validation samples (optional)
        stage_config: Stage configuration
        stage_name: Stage name for logging
        weight_decay: Weight decay for optimizer
        gradient_clip_norm: Gradient clipping norm
        loss_direction_weight: Direction loss weight
        loss_return_weight: Return loss weight
        direction_classes: Number of direction classes
        use_amp: Whether to use automatic mixed precision
        accumulation_steps: Gradient accumulation steps
        device: Device to train on
        val_samples: Optional validation samples for early stopping
        early_stopping_patience: Epochs without improvement before stopping

    Returns:
        Dict with stage results
    """
    from torch.cuda.amp import GradScaler, autocast

    # Apply freeze mask
    _apply_freeze_mask(model, stage_config)

    # Get only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"{stage_name}: no trainable parameters after freeze mask")

    logger.info(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=stage_config.learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=early_stopping_patience, factor=0.5
    )

    # V4Loss + per-stage weights (replaces the old huber + per-sample
    # classification_criterion combo). Direction loss is routed inside
    # _compute_v4_loss_for_buckets; loss_return_weight is now absorbed into
    # the V4LossWeights component weights (kept in the signature for caller
    # compatibility).
    v4_loss = V4Loss(_v4_loss_weights_for_stage(stage_config.stage))

    # Pre-bucket samples by as_of_date once (time-sorted, no future leakage).
    train_buckets = _group_samples_by_date(train_samples)
    val_buckets = _group_samples_by_date(val_samples) if val_samples else []

    # AMP scaler
    scaler = GradScaler() if use_amp and device == "cuda" else None

    epoch_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state: dict[str, Any] | None = None

    for epoch in range(stage_config.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, bucket in enumerate(train_buckets):
            model.train()
            if step % 5 == 0:
                logger.info(
                    f"  epoch {epoch + 1}/{stage_config.epochs} step {step}/{len(train_buckets)}"
                )

            # Forward + V4Loss for the whole date bucket.
            with autocast(enabled=scaler is not None):
                loss, _ = _compute_v4_loss_for_buckets(
                    model,
                    train_samples,
                    [bucket],
                    v4_loss,
                    enable_seat_branch=(not stage_config.freeze_seat),
                    direction_classes=direction_classes,
                    direction_weight=loss_direction_weight,
                    device=device,
                )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            # Backward pass with AMP
            if scaler is not None:
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call,unused-ignore]
            else:
                loss.backward()  # type: ignore[no-untyped-call,unused-ignore]

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                if gradient_clip_norm is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            epoch_loss += float(loss.detach().cpu()) * accumulation_steps * len(bucket)

        # Average training loss (per sample, comparable across bucket sizes)
        avg_train_loss = epoch_loss / max(len(train_samples), 1)
        epoch_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = float("inf")
        if val_samples:
            model.eval()
            with torch.no_grad():
                val_loss, _ = _compute_v4_loss_for_buckets(
                    model,
                    val_samples,
                    val_buckets,
                    v4_loss,
                    enable_seat_branch=(not stage_config.freeze_seat),
                    direction_classes=direction_classes,
                    direction_weight=loss_direction_weight,
                    device=device,
                )
                avg_val_loss = float(val_loss.detach().cpu())
            val_losses.append(avg_val_loss)
            logger.info(
                f"  Epoch {epoch + 1}/{stage_config.epochs}: "
                f"Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}"
            )
        else:
            logger.info(f"  Epoch {epoch + 1}/{stage_config.epochs}: Loss = {avg_train_loss:.4f}")

        # Use validation loss for scheduler if available, else training loss
        scheduler_metric = avg_val_loss if val_samples else avg_train_loss
        scheduler.step(scheduler_metric)

        # Check for non-finite loss
        if not (scheduler_metric < float("inf") and scheduler_metric > float("-inf")):
            raise ValueError(f"{stage_name} produced non-finite loss at epoch {epoch}")

        # Early stopping on validation loss. The first epoch must also update
        # best_val_loss (initialized to +inf), otherwise single-epoch stages
        # report Loss: inf and never capture a best model state.
        if val_samples:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  New best validation loss: {best_val_loss:.4f}")
            elif len(val_losses) > 1:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break

    # Restore best model if we have validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
        logger.info(f"  Restored best model with val_loss = {best_val_loss:.4f}")

    # Unfreeze for next stage
    for param in model.parameters():
        param.requires_grad = True

    return {
        "loss": best_val_loss if val_samples else avg_train_loss,
        "epochs": epoch + 1,  # Actual epochs trained
        "loss_history": epoch_losses,
        "val_loss_history": val_losses if val_samples else None,
    }


def _apply_freeze_mask(
    model: DTSeatHGTv3Model,
    stage_config: DTSeatHGTv3TrainingStageConfig,
) -> None:
    """Apply freeze mask based on training stage.

    Model components:
    - Base: stock_gru, stock_static_mlp, base_projection, base_relation_branch,
            industry_projection, layer_norm, base_rank_head, base_return_head
    - Seat: seat_projection, seat_relation_branch, seat_rank_delta_head,
            seat_return_delta_head, seat_gate_mlp, seat_style_encoder
    - Final: final_rank_head, final_return_head, direction_head, uncertainty_head, dropout

    The base embedding (and everything feeding it: stock_gru, stock_static_mlp,
    base_projection, industry_projection, layer_norm, base_relation_branch) must be
    frozen together so that Stage B cannot move the base representation whose detached
    prediction forms the residual target.
    """
    # First, unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    if stage_config.freeze_base:
        # Freeze all base/embedding components
        for component in (
            model.stock_gru,
            model.stock_static_mlp,
            model.base_projection,
            model.base_relation_branch,
            model.industry_projection,
            model.layer_norm,
            model.base_rank_head,
            model.base_return_head,
        ):
            for param in component.parameters():
                param.requires_grad = False

    if stage_config.freeze_seat:
        # Freeze all seat-related components
        for seat_component in (
            model.seat_projection,
            model.seat_relation_branch,
            model.seat_rank_delta_head,
            model.seat_return_delta_head,
            model.seat_gate_mlp,
            model.seat_style_encoder,
        ):
            for param in seat_component.parameters():
                param.requires_grad = False


def default_v3_training_config() -> DTSeatHGTv3TrainingConfig:
    """Create default three-stage training configuration for R5."""
    return DTSeatHGTv3TrainingConfig(
        model=DTSeatHGTv3Config(
            hidden_dim=32,
            stock_gru_hidden_dim=24,
            stock_gru_layers=1,
            stock_static_hidden_dim=16,
            stock_sequence_length=20,
            stock_sequence_feature_dim=5,
            seat_style_count=5,
            seat_style_embedding_dim=16,
            seat_feature_dim=32,
            num_heads=2,
            base_relation_layers=1,
            seat_relation_layers=1,
            dropout=0.1,
            relation_dropout=0.1,
            seat_gate_initial_bias=-2.0,
        ),
        stage1=DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.BASE_ONLY,
            epochs=3,
            learning_rate=0.001,
            freeze_base=False,
            freeze_seat=True,
        ),
        stage2=DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.SEAT_RESIDUAL,
            epochs=2,
            learning_rate=0.0005,
            freeze_base=True,
            freeze_seat=False,
        ),
        stage3=DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.JOINT_FINETUNE,
            epochs=2,
            learning_rate=0.0001,  # 0.1x Stage A LR
            freeze_base=False,
            freeze_seat=False,
        ),
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        seeds=(2026, 2027, 2028),  # Fixed seeds for R5
        loss_direction_weight=1.0,
        loss_return_weight=1.0,
        direction_classes=2,
        use_amp=True,
        accumulation_steps=1,
    )


def resume_dt_seat_hgt_v3(
    checkpoint_path: str | Path,
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTv3TrainingConfig | None = None,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> DTSeatHGTv3TrainingResult:
    """Resume training from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        samples: Training samples
        config: Optional config override (uses checkpoint config if None)
        device: Device to train on
        output_root: Output directory for new checkpoints

    Returns:
        Training result with stage metrics
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise PyGModelUnavailableError(
            "DT-SeatHGT v3 training requires PyTorch."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint or use provided
    if config is None:
        model_config_dict = checkpoint.get("config", {})
        model_config = DTSeatHGTv3Config(**model_config_dict)
        config = default_v3_training_config()
        config = DTSeatHGTv3TrainingConfig(
            model=model_config,
            stage1=config.stage1,
            stage2=config.stage2,
            stage3=config.stage3,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            seeds=(checkpoint.get("seed", 2026),),
            loss_direction_weight=config.loss_direction_weight,
            loss_return_weight=config.loss_return_weight,
            direction_classes=config.direction_classes,
            use_amp=config.use_amp,
            accumulation_steps=config.accumulation_steps,
        )

    # Build model and load weights
    model = build_dt_seat_hgt_v3(
        config.model, cast("dict[str, Any] | None", samples[0].graph)
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Continue training from next stage (if stages were partially completed)
    # For simplicity, we retrain all stages from the checkpoint state
    logger.info(f"Resuming training from {checkpoint_path}")

    # Train all stages (in a full implementation, would track which stage to resume from)
    stage_results: dict[str, dict[str, Any]] = {}
    total_epochs = 0
    final_loss = 0.0

    for stage_config in (config.stage1, config.stage2, config.stage3):
        stage_name = stage_config.stage.name
        stage_result = _train_single_stage(
            model=model,
            train_samples=samples,
            stage_config=stage_config,
            stage_name=stage_name,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            loss_direction_weight=config.loss_direction_weight,
            loss_return_weight=config.loss_return_weight,
            direction_classes=config.direction_classes,
            use_amp=config.use_amp,
            accumulation_steps=config.accumulation_steps,
            device=device,
        )
        stage_results[stage_name] = stage_result
        total_epochs += stage_result["epochs"]
        final_loss = stage_result["loss"]
        logger.info(f"Resumed {stage_name}: Loss = {final_loss:.4f}")

    # Save new checkpoint
    output_path = Path(output_root) if output_root else checkpoint_path.parent
    seed_suffix = checkpoint.get('seed', 'unknown')
    new_checkpoint_path = output_path / f"dtseathgt_v3_resumed_{seed_suffix}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.model.to_dict(),
            "run_id": checkpoint.get("run_id", "resumed") + "_resumed",
            "stage_results": stage_results,
            "seed": checkpoint.get("seed", 2026),
            "resumed_from": str(checkpoint_path),
        },
        new_checkpoint_path,
    )

    return DTSeatHGTv3TrainingResult(
        run_id=checkpoint.get("run_id", "resumed") + "_resumed",
        status="completed",
        model_valid=False,
        checkpoint_path=new_checkpoint_path,
        stage_results=stage_results,
        total_epochs=total_epochs,
        final_loss=final_loss,
        seed=checkpoint.get("seed", 2026),
        gpu_memory_peak=None,
    )
