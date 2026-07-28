"""Loss functions for DT-SeatHGT v3 three-stage training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DTSeatHGTv3LossWeights:
    """Loss weights for DT-SeatHGT v3."""

    # Return prediction
    return_smooth_l1: float = 1.0

    # Ranking
    ranking_pairwise: float = 0.5

    # Direction
    direction_weight: float = 1.0

    # Uncertainty
    uncertainty_nll: float = 0.1

    # Seat gate
    gate_sparsity: float = 0.01

    # Style regularization
    style_conditional_entropy: float = 0.1
    style_marginal_balance: float = 0.1
    style_temporal_consistency: float = 0.05

    def __post_init__(self) -> None:
        for weight in (
            self.return_smooth_l1,
            self.ranking_pairwise,
            self.direction_weight,
            self.uncertainty_nll,
            self.gate_sparsity,
            self.style_conditional_entropy,
            self.style_marginal_balance,
            self.style_temporal_consistency,
        ):
            if weight < 0:
                raise ValueError("Loss weights must be non-negative")


def smooth_l1_return_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 loss for return prediction.

    Args:
        predictions: [N] return predictions
        targets: [N] true returns
        indices: [batch_size] indices into predictions/targets

    Returns:
        Scalar loss
    """
    pred = predictions[indices]
    tgt = targets[indices]
    return F.smooth_l1_loss(pred, tgt)


def sampled_pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    sample_size: int = 64,
) -> torch.Tensor:
    """Sampled pairwise ranking loss (bounded).

    Only samples a subset of pairs to avoid O(N²) computation.

    Args:
        predictions: [N] predictions
        targets: [N] labels for ranking
        indices: [batch_size] indices into predictions/targets
        sample_size: Maximum pairs to sample

    Returns:
        Scalar loss
    """
    pred = predictions[indices]
    tgt = targets[indices]

    batch_size = pred.size(0)
    num_pairs = min(sample_size, batch_size * (batch_size - 1) // 2)

    # Sample random pairs
    with torch.no_grad():
        i = torch.randint(0, batch_size, (num_pairs,), device=pred.device)
        j = torch.randint(0, batch_size, (num_pairs,), device=pred.device)
        i, j = torch.where(i != j)

        if i.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

    # Pairwise differences
    pred_diff = pred[i] - pred[j]
    tgt_diff = tgt[i] - tgt[j]

    # Margin-based ranking loss
    margin = 0.1
    loss = torch.relu(margin - tgt_diff.sign() * pred_diff).mean()

    return loss


def pearson_correlation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Negative Pearson correlation loss.

    Args:
        predictions: [N] predictions
        targets: [N] targets
        indices: [batch_size] indices

    Returns:
        Scalar loss (negative correlation = higher loss)
    """
    pred = predictions[indices]
    tgt = targets[indices]

    # Center the variables
    pred_centered = pred - pred.mean()
    tgt_centered = tgt - tgt.mean()

    # Pearson correlation
    numerator = (pred_centered * tgt_centered).sum()
    denominator = torch.sqrt(
        (pred_centered ** 2).sum() * (tgt_centered ** 2).sum()
    )
    pearson = numerator / (denominator + 1e-8)

    # Negative because we want to minimize
    return -pearson


def direction_loss(
    direction_logits: torch.Tensor,
    direction_labels: torch.Tensor,
    indices: torch.Tensor,
    direction_classes: int = 2,
) -> torch.Tensor:
    """Direction classification loss.

    Args:
        direction_logits: [N] or [N, num_classes] logits
        direction_labels: [N] labels (-1=down, 0=neutral, 1=up or binary)
        indices: [batch_size] indices
        direction_classes: 2 for binary BCE, >2 for multi-class CE

    Returns:
        Scalar loss
    """
    logits = direction_logits[indices]
    labels = direction_labels[indices]

    if direction_classes == 2:
        # Binary: BCEWithLogitsLoss
        # Convert labels: -1/0/1 -> 0/1 for BCE
        binary_labels = (labels > 0).float()
        return F.binary_cross_entropy_with_logits(logits, binary_labels)
    else:
        # Multi-class: CrossEntropyLoss
        # Map: -1 -> 0, 0 -> 1, 1 -> 2
        class_labels = labels.long() + 1
        return F.cross_entropy(logits, class_labels)


def uncertainty_nll_loss(
    uncertainty: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Uncertainty negative log-likelihood loss.

    Models heteroscedastic uncertainty.

    Args:
        uncertainty: [N] predicted uncertainty (softplus activated)
        predictions: [N] predictions
        targets: [N] targets
        indices: [batch_size] indices

    Returns:
        Scalar loss
    """
    unc = uncertainty[indices]
    pred = predictions[indices]
    tgt = targets[indices]

    # Gaussian NLL: -log p(y|f) ∝ 0.5 * log(sigma²) + (y-f)² / (2*sigma²)
    log_variance = torch.log(unc + 1e-6)
    squared_error = (pred - tgt) ** 2
    nll = 0.5 * log_variance + squared_error / (2 * unc ** 2 + 1e-6)

    return nll.mean()


def gate_sparsity_loss(gate_values: torch.Tensor) -> torch.Tensor:
    """Encourage sparse seat gates.

    Args:
        gate_values: [N] gate values (0 to 1)

    Returns:
        Scalar loss (encourage gates toward 0)
    """
    # L1 regularization on gates
    return gate_values.mean()


def compute_total_loss(
    outputs: dict[str, Any],
    sample: Any,
    weights: DTSeatHGTv3LossWeights,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total loss for a training stage.

    Args:
        outputs: Model forward outputs
        sample: Training sample with labels
        weights: Loss weights
        stage: Training stage ("base", "seat", "joint")

    Returns:
        Tuple of (total_loss, loss_components)
    """
    losses: dict[str, torch.Tensor] = {}

    # Extract predictions
    final_return = outputs["final_return_prediction"]
    base_return = outputs["base_return_prediction"]
    direction_logits = outputs["direction_logit"]
    uncertainty = outputs["uncertainty"]
    seat_gate = outputs["seat_gate"]

    # Sample indices (for single-sample batch, this is just index 0)
    target_idx = sample.target_node_index
    target_idx = (
        target_idx
        if isinstance(target_idx, torch.Tensor)
        else torch.tensor([target_idx], device=final_return.device)
    )

    # Get labels
    true_return = torch.tensor(
        [sample.returns.get(1, 0.0)], dtype=torch.float32, device=final_return.device
    )
    direction_label = torch.tensor(
        [sample.labels.get(1, 0)], dtype=torch.float32, device=direction_logits.device
    )

    # Determine which return prediction to use
    if stage == "base":
        # Stage A: Only base prediction
        return_pred = base_return[target_idx]
    elif stage == "seat":
        # Stage B: Use final (with detached base for residual target)
        # The residual target is: true - detach(base)
        return_pred = final_return[target_idx]
    else:
        # Stage C: Joint - use final
        return_pred = final_return[target_idx]

    # Return loss
    if weights.return_smooth_l1 > 0:
        losses["return_smooth_l1"] = smooth_l1_return_loss(
            return_pred,
            true_return.expand_as(return_pred),
            torch.arange(len(return_pred), device=return_pred.device),
        ) * weights.return_smooth_l1

    # Ranking loss
    if weights.ranking_pairwise > 0:
        losses["ranking_pairwise"] = sampled_pairwise_ranking_loss(
            return_pred,
            true_return.expand_as(return_pred),
            torch.arange(len(return_pred), device=return_pred.device),
        ) * weights.ranking_pairwise

    # Direction loss
    if weights.direction_weight > 0:
        losses["direction"] = direction_loss(
            direction_logits,
            direction_label,
            torch.arange(len(direction_logits), device=direction_logits.device),
        ) * weights.direction_weight

    # Uncertainty NLL
    if weights.uncertainty_nll > 0:
        losses["uncertainty_nll"] = uncertainty_nll_loss(
            uncertainty,
            return_pred,
            true_return.expand_as(return_pred),
            torch.arange(len(return_pred), device=return_pred.device),
        ) * weights.uncertainty_nll

    # Gate sparsity
    if weights.gate_sparsity > 0:
        losses["gate_sparsity"] = gate_sparsity_loss(seat_gate) * weights.gate_sparsity

    # Style regularization
    if "seat_style_probs" in outputs and outputs["seat_style_probs"] is not None:
        style_probs = outputs["seat_style_probs"]

        if weights.style_conditional_entropy > 0:
            entropy = -(style_probs * torch.log(style_probs + 1e-8)).sum(dim=-1)
            losses["style_conditional_entropy"] = entropy.mean() * weights.style_conditional_entropy

        if weights.style_marginal_balance > 0:
            mean_usage = style_probs.mean(dim=0)
            target_usage = 1.0 / style_probs.size(-1)
            balance = ((mean_usage - target_usage) ** 2).mean()
            losses["style_marginal_balance"] = (
                balance * weights.style_marginal_balance
            )

    # Total loss
    total_loss = sum(losses.values())

    return cast(torch.Tensor, total_loss), losses
