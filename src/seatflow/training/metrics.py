from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    sample_count: int
    accuracy: float
    positive_rate: float
    mean_probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "positive_rate": self.positive_rate,
            "mean_probability": self.mean_probability,
        }


def binary_classification_metrics(
    probabilities: tuple[float, ...],
    labels: tuple[int, ...],
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("at least one prediction is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    for probability in probabilities:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must be finite values in [0, 1]")
    for label in labels:
        if label not in (0, 1):
            raise ValueError("labels must be binary integers")

    predictions = tuple(1 if probability >= threshold else 0 for probability in probabilities)
    correct = sum(
        1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label
    )
    return BinaryClassificationMetrics(
        sample_count=len(probabilities),
        accuracy=round(correct / len(probabilities), 6),
        positive_rate=round(sum(labels) / len(labels), 6),
        mean_probability=round(sum(probabilities) / len(probabilities), 6),
    )
