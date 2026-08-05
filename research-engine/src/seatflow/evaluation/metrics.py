"""Evaluation metrics for SeatFlow V3.

Implements:
- MAE, MSE, RMSE, Huber
- AUC, Brier
- IC, RankIC, ICIR, RankICIR
- NDCG@K, Hit@K
- Edge AUPR, Recall@K
- ECE (Expected Calibration Error)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss element-wise then mean."""
    diff = np.abs(y_true - y_pred)
    quadratic = np.minimum(diff, delta)
    linear = diff - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for binary classification (y_true in {0, 1})."""
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    n = len(y_true)
    if n == 0:
        return 0.5
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney statistic
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities)."""
    return float(np.mean((y_true - y_prob) ** 2))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Rank data, average ties."""
    from scipy.stats import rankdata as scipy_rankdata
    return scipy_rankdata(x, method="average")


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson IC (Information Coefficient)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman RankIC."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    r_true = _rankdata(y_true)
    r_pred = _rankdata(y_pred)
    if np.std(r_true) < 1e-12 or np.std(r_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def icir(ic_values: list[float]) -> float:
    """ICIR = mean(IC) / std(IC) across multiple periods/seeds."""
    if len(ic_values) < 2:
        return 0.0
    arr = np.array(ic_values)
    if arr.std() < 1e-12:
        return 0.0
    return float(arr.mean() / arr.std())


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    # Top-k by predicted score
    top_k_idx = np.argsort(-y_pred)[:k]
    ideal_idx = np.argsort(-y_true)[:k]
    dcg = sum(y_true[i] / math.log2(j + 2) for j, i in enumerate(top_k_idx))
    idcg = sum(y_true[i] / math.log2(j + 2) for j, i in enumerate(ideal_idx))
    if idcg < 1e-12:
        return 0.0
    return float(dcg / idcg)


def hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Hit@K: proportion of top-K predictions matching top-K truths."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    top_k_pred = set(np.argsort(-y_pred)[:k])
    top_k_true = set(np.argsort(-y_true)[:k])
    return float(len(top_k_pred & top_k_true) / k)


def edge_aupr(
    edge_true: np.ndarray,
    edge_score: np.ndarray,
) -> float:
    """Area Under Precision-Recall curve for edge prediction."""
    edge_true = np.asarray(edge_true).flatten()
    edge_score = np.asarray(edge_score).flatten()
    n = len(edge_true)
    if n == 0 or edge_true.sum() == 0:
        return 0.0
    # Sort by score descending
    order = np.argsort(-edge_score)
    sorted_true = edge_true[order]
    tp_cumsum = np.cumsum(sorted_true)
    recall = tp_cumsum / edge_true.sum()
    precision = tp_cumsum / np.arange(1, n + 1)
    # Use sklearn's average_precision_score for exactness
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(edge_true, edge_score))
    except ImportError:
        # Fallback: trapezoidal integration
        # Precision-recall curve
        prev_recall = 0.0
        aupr = 0.0
        for r, p in zip(recall, precision):
            aupr += p * (r - prev_recall)
            prev_recall = r
        return float(aupr)


def recall_at_k(
    edge_true: np.ndarray,
    edge_score: np.ndarray,
    k: int,
) -> float:
    """Recall@K for edge prediction."""
    edge_true = np.asarray(edge_true).flatten()
    edge_score = np.asarray(edge_score).flatten()
    n = len(edge_true)
    if n == 0 or edge_true.sum() == 0:
        return 0.0
    k = min(k, n)
    top_k_idx = np.argsort(-edge_score)[:k]
    hits = edge_true[top_k_idx].sum()
    return float(hits / edge_true.sum())


def ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error."""
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    if len(y_true) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > lo) & (y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        ece_val += abs(bin_acc - bin_conf) * in_bin.sum() / n
    return float(ece_val)


def uncertainty_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    target_coverage: float = 0.9,
) -> float:
    """Fraction of predictions where actual error is within the predicted uncertainty band."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    uncertainty = np.asarray(uncertainty).flatten()
    if len(y_true) == 0:
        return 0.0
    abs_error = np.abs(y_true - y_pred)
    threshold = np.quantile(uncertainty, target_coverage)
    covered = (abs_error <= threshold).mean()
    return float(covered)


# ── Multi-period metric aggregation ──────────────────────────────────


def aggregate_metric(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max for a metric across multiple periods."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
