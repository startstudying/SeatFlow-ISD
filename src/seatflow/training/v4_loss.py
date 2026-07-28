"""DT-SeatHGT v4 loss (independent module, not yet wired into training).

Design (see docs/audit/07_LOSS_REDESIGN.md and docs/audit/MASTER_ROADMAP.md):

1. rank-based listwise loss  -- a vectorized, numerically-stable ListMLE
   (simplified Plackett-Luce top-1 likelihood). Replaces the high-variance
   random-pairwise ranking loss used in v3 (`sampled_pairwise_ranking_loss`).
   ListMLE considers the full list order per cross section, so gradients are
   deterministic and lower variance than sampled pairs.

2. residual-aware loss        -- the base return and the seat residual are
   decoupled so the seat branch never backprops into the base branch. The
   seat residual target is ``target_return - base_pred.detach()`` (mirrors
   the V4-A8 fix in dtseathgt_v3.py Stage B). The base branch is trained
   directly against the full return.

Public surface:
    - V4LossWeights           : frozen dataclass holding component weights
    - listwise_rank_loss(...) : standalone ListMLE (used standalone or by V4Loss)
    - residual_seat_loss(...) : standalone residual-aware seat loss
    - V4Loss                  : callable combining the two into a dict

Constraints (YAGNI, per project rules):
    - No new dependencies; torch only.
    - No Mamba / hypergraph / LLM / complex modules.
    - Not wired into the main training loop; integration is the next phase.

The returned dict always contains the scalar keys: ``loss``, ``base_loss``,
``seat_loss``, ``listwise_loss``. Optional components (direction / gate) are
added only when the corresponding inputs are provided.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Numerical stability / sentinel helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8  # small constant for log / division guards


def _suffix_logsumexp(scores: torch.Tensor) -> torch.Tensor:
    """Suffix log-sum-exp: out[i] = logsumexp(scores[i:]).

    Implemented via ``torch.logcumsumexp`` on a reversed tensor so it is
    numerically stable for large score magnitudes. ``scores`` is 1-D.

    ``out[i] = log( sum_{k>=i} exp(scores[k]) )``.
    """
    if scores.numel() == 0:
        return scores
    reversed_scores = torch.flip(scores, dims=[0])
    cumulative = torch.logcumsumexp(reversed_scores, dim=0)
    return torch.flip(cumulative, dims=[0])


def _group_indices(dates: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    """Group flat indices by date (cross-section key).

    Single shared implementation to avoid duplicating the per-date grouping
    that v3 sprinkled across ``ranking_losses.py`` / ``hard_sample_mining.py``
    (audit finding: "日期分组逻辑重复").
    """
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, value in enumerate(dates):
        grouped[value].append(index)
    return grouped


def _as_index_tensor(indices: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# ListMLE: rank-based listwise loss
# ---------------------------------------------------------------------------


def listwise_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dates: Sequence[Hashable] | None = None,
) -> torch.Tensor:
    """ListMLE ranking loss (simplified Plackett-Luce top-1 likelihood).

    For a single cross-section list of length ``n``:

        L = -sum_{j=1}^{n} s_{(j)} + sum_{j=1}^{n} logsumexp( s_{(j)}, ..., s_{(n)} )

    where ``s_{(j)}`` is the prediction at the j-th position when the list is
    sorted by the *true* target in descending order. Equivalently, this is the
    negative log-likelihood of the true ranking under the Plackett-Luce model.

    Properties enforced by the implementation (covered by tests):
        - Rank consistency: a prediction permutation that matches the true
          ranking yields a strictly lower loss than a reversed one.
        - Determinism: no random sampling -> identical inputs give identical
          gradients (unlike v3 ``sampled_pairwise_ranking_loss``).
        - Numerical stability via ``logcumsumexp`` (no overflow on large scores).

    Args:
        predictions: 1-D tensor of scores, shape ``[N]``.
        targets:     1-D tensor of ranking labels, shape ``[N]``.
        dates:       optional cross-section keys aligned with ``predictions``.
            When provided, ListMLE is computed independently per date group and
            averaged. When ``None`` the whole tensor is treated as one list.

    Returns:
        Scalar tensor. Zero (with gradient connection) when no group has >=2
        elements.
    """
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("listwise_rank_loss expects 1-D predictions and targets")
    if predictions.shape != targets.shape:
        raise ValueError("listwise_rank_loss predictions and targets must align")

    if dates is None:
        return _listwise_rank_loss_single(predictions, targets)

    if len(dates) != int(predictions.numel()):
        raise ValueError("listwise_rank_loss dates must align with predictions")

    losses: list[torch.Tensor] = []
    for indices in _group_indices(dates).values():
        if len(indices) < 2:
            continue  # ranking is undefined for a single-element list
        idx = _as_index_tensor(indices, predictions.device)
        group_pred = predictions.index_select(0, idx)
        group_tgt = targets.index_select(0, idx)
        losses.append(_listwise_rank_loss_single(group_pred, group_tgt))
    if not losses:
        # Preserve the autograd graph: 0 * sum keeps grad_fn connected.
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


def _listwise_rank_loss_single(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ListMLE for a single cross-section list (length >= 2 assumed by caller)."""
    # Order the list by true target, descending. Stable sort keeps ties in
    # their original order so the loss is well-defined on equal targets.
    sorted_idx = torch.argsort(targets, descending=True, stable=True)
    sorted_scores = scores.index_select(0, sorted_idx)

    # suffix_logsumexp(sorted_scores)[j] = logsumexp(sorted_scores[j:])
    # L = sum_j ( logsumexp(sorted_scores[j:]) - sorted_scores[j] )
    suffix = _suffix_logsumexp(sorted_scores)
    loss = (suffix - sorted_scores).mean()
    return loss


# ---------------------------------------------------------------------------
# Residual-aware seat loss
# ---------------------------------------------------------------------------


def residual_seat_loss(
    base_pred: torch.Tensor,
    seat_residual_pred: torch.Tensor,
    target_return: torch.Tensor,
) -> torch.Tensor:
    """Residual-aware loss for the seat branch.

    The seat residual target is the part of the return the base model cannot
    explain: ``target_return - base_pred.detach()``. Detaching ``base_pred``
    means the seat branch never backprops into the base branch (audit P0
    finding #8 "Stage B No Residual Target"; V4-A8 fix). The seat head is thus
    trained to predict the *residual* signal alone.

    Args:
        base_pred:          1-D tensor, base model return prediction.
        seat_residual_pred: 1-D tensor, seat residual prediction (already gated
            by the caller if a gate is used).
        target_return:      1-D tensor, true return.

    Returns:
        Scalar SmoothL1 loss between ``seat_residual_pred`` and the detached
        residual target.
    """
    if base_pred.ndim != 1 or seat_residual_pred.ndim != 1 or target_return.ndim != 1:
        raise ValueError("residual_seat_loss expects 1-D tensors")
    if not (base_pred.shape == seat_residual_pred.shape == target_return.shape):
        raise ValueError("residual_seat_loss tensors must align in shape")

    residual_target = target_return - base_pred.detach()
    return F.smooth_l1_loss(seat_residual_pred, residual_target)


# ---------------------------------------------------------------------------
# Combined V4Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V4LossWeights:
    """Weights for the v4 loss components.

    All non-negative. ``base_weight`` and ``seat_weight`` should both be
    positive so both branches receive signal; ``listwise_weight`` may be zero
    to disable ranking.
    """

    base_weight: float = 1.0
    seat_weight: float = 1.0
    listwise_weight: float = 0.5
    direction_weight: float = 0.0
    gate_sparsity_weight: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("base_weight", self.base_weight),
            ("seat_weight", self.seat_weight),
            ("listwise_weight", self.listwise_weight),
            ("direction_weight", self.direction_weight),
            ("gate_sparsity_weight", self.gate_sparsity_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.base_weight == 0 and self.seat_weight == 0 and self.listwise_weight == 0:
            raise ValueError(
                "at least one of base_weight / seat_weight / listwise_weight must be positive"
            )


class V4Loss:
    """Combined v4 loss: listwise ranking + residual-aware base/seat.

    Callable interface::

        out = V4Loss(weights)(
            base_pred=...,            # [N]
            seat_residual_pred=...,   # [N]
            target_return=...,        # [N]
            dates=...,                # optional, length N
            direction_logits=...,     # optional, [N] or [N, C]
            direction_labels=...,     # optional, [N]
            seat_gate=...,            # optional, [N]
        )

    Returns a dict with keys (all scalar tensors):
        - ``loss``           : total weighted loss
        - ``base_loss``      : SmoothL1(base_pred, target_return)
        - ``seat_loss``      : residual_seat_loss(...)  (decoupled)
        - ``listwise_loss``  : ListMLE on the full prediction (base + seat)
        - ``direction_loss`` : BCE/CE (only if direction_* provided)
        - ``gate_loss``      : gate L1 sparsity (only if seat_gate provided)

    The "full prediction" used for ListMLE is ``base_pred + seat_residual_pred``
    (i.e. the final return the model emits). Ranking is computed on the
    composite because the downstream evaluation ranks stocks by final return.
    """

    def __init__(self, weights: V4LossWeights | None = None) -> None:
        self.weights = weights if weights is not None else V4LossWeights()

    def __call__(
        self,
        base_pred: torch.Tensor,
        seat_residual_pred: torch.Tensor,
        target_return: torch.Tensor,
        dates: Sequence[Hashable] | None = None,
        *,
        direction_logits: torch.Tensor | None = None,
        direction_labels: torch.Tensor | None = None,
        direction_classes: int = 2,
        seat_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        w = self.weights

        if base_pred.ndim != 1 or seat_residual_pred.ndim != 1 or target_return.ndim != 1:
            raise ValueError("base_pred / seat_residual_pred / target_return must be 1-D")
        if not (base_pred.shape == seat_residual_pred.shape == target_return.shape):
            raise ValueError("base_pred, seat_residual_pred, target_return must align")

        # Base branch: trained directly on the full return.
        base_loss = F.smooth_l1_loss(base_pred, target_return)

        # Seat branch: residual-aware, base contribution detached inside.
        seat_loss = residual_seat_loss(base_pred, seat_residual_pred, target_return)

        # Ranking on the composite final return.
        full_pred = base_pred + seat_residual_pred
        listwise_loss = listwise_rank_loss(full_pred, target_return, dates)

        components: dict[str, torch.Tensor] = {
            "base_loss": base_loss,
            "seat_loss": seat_loss,
            "listwise_loss": listwise_loss,
        }

        total = (
            w.base_weight * base_loss
            + w.seat_weight * seat_loss
            + w.listwise_weight * listwise_loss
        )

        if direction_logits is not None and direction_labels is not None and w.direction_weight > 0:
            direction_loss = _direction_loss(
                direction_logits, direction_labels, direction_classes
            )
            components["direction_loss"] = direction_loss
            total = total + w.direction_weight * direction_loss

        if seat_gate is not None and w.gate_sparsity_weight > 0:
            gate_loss = seat_gate.mean()
            components["gate_loss"] = gate_loss
            total = total + w.gate_sparsity_weight * gate_loss

        components["loss"] = total
        return components


def _direction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    direction_classes: int,
) -> torch.Tensor:
    """Direction classification loss (BCE for binary, CE for multi-class).

    Mirrors ``dt_seat_hgt_v3_losses.direction_loss`` but without the index
    indirection (v4 passes already-aligned tensors).
    """
    if direction_classes == 2:
        binary_labels = (labels > 0).float()
        if logits.ndim == 2:
            logits = logits.squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, binary_labels)
    # Multi-class: map -1/0/1 -> 0/1/2
    class_labels = labels.long() + 1
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    return F.cross_entropy(logits, class_labels)
