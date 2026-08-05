"""Calibration metrics (Task 18)."""

from __future__ import annotations

import numpy as np

from seatflow.evaluation.metrics import ece, brier


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute reliability diagram data.

    Returns:
        Dict with:
        - bin_centers: midpoints of probability bins
        - bin_accuracies: observed accuracy in each bin
        - bin_counts: number of samples in each bin
        - bin_confidences: mean predicted probability in each bin
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    n = len(y_true)
    if n == 0:
        return {
            "bin_centers": np.array([]),
            "bin_accuracies": np.array([]),
            "bin_counts": np.array([]),
            "bin_confidences": np.array([]),
        }

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    bin_confidences = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > lo) & (y_prob <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accuracies.append(y_true[in_bin].mean())
        bin_counts.append(count)
        bin_confidences.append(y_prob[in_bin].mean())

    return {
        "bin_centers": np.array(bin_centers),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_counts": np.array(bin_counts),
        "bin_confidences": np.array(bin_confidences),
    }


def calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute multiple calibration metrics.

    Returns:
        Dict with ECE, Brier score, and maximum calibration error.
    """
    ece_val = ece(y_true, y_prob, n_bins=n_bins)
    brier_val = brier(y_true, y_prob)

    # Maximum calibration error
    diag = reliability_diagram(y_true, y_prob, n_bins=n_bins)
    if len(diag["bin_accuracies"]) > 0:
        mce = float(
            np.max(np.abs(diag["bin_accuracies"] - diag["bin_confidences"]))
        )
    else:
        mce = 0.0

    return {
        "ece": ece_val,
        "brier": brier_val,
        "mce": mce,
    }


def uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, float]:
    """Evaluate uncertainty calibration for regression predictions.

    A well-calibrated model should have uncertainty proportional to
    actual error.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    uncertainty = np.asarray(uncertainty).flatten()

    if len(y_true) == 0:
        return {"coverage_90": 0.0, "correlation": 0.0, "nll": 0.0}

    abs_error = np.abs(y_true - y_pred)

    # Coverage at 90%: fraction of errors within 1.645 * uncertainty (z-score)
    threshold_90 = 1.645 * uncertainty
    coverage_90 = float((abs_error <= threshold_90).mean())

    # Correlation between predicted uncertainty and actual error
    if np.std(uncertainty) > 1e-12 and np.std(abs_error) > 1e-12:
        corr = float(np.corrcoef(uncertainty, abs_error)[0, 1])
    else:
        corr = 0.0

    # Negative log-likelihood (Gaussian)
    # NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2
    sigma = np.maximum(uncertainty, 1e-8)
    nll = float(
        np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * (abs_error / sigma) ** 2)
    )

    return {
        "coverage_90": coverage_90,
        "correlation": corr,
        "nll": nll,
    }
