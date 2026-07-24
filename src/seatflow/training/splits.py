from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from seatflow.datasets.windows import WindowSample
from seatflow.training.preflight import TimeSplit, validate_time_split


@dataclass(frozen=True)
class SplitPolicy:
    purge_days: int
    embargo_days: int
    max_horizon_days: int

    def __post_init__(self) -> None:
        if self.purge_days < 0:
            raise ValueError("purge_days must be non-negative")
        if self.embargo_days < self.max_horizon_days:
            raise ValueError("embargo_days must be at least max_horizon_days")
        if self.max_horizon_days <= 0:
            raise ValueError("max_horizon_days must be positive")


def assign_fixed_time_split(
    samples: tuple[WindowSample, ...],
    split: TimeSplit,
    policy: SplitPolicy,
) -> dict[str, tuple[str, ...]]:
    errors = validate_time_split(split)
    if errors:
        raise ValueError("; ".join(errors))

    assignments: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for sample in sorted(samples, key=lambda item: (item.as_of_date, item.sample_id)):
        label_end = max(label.label_end_date for label in sample.labels)
        if _in_train(sample.as_of_date, label_end, split, policy):
            assignments["train"].append(sample.sample_id)
        elif _in_valid(sample.as_of_date, label_end, split, policy):
            assignments["valid"].append(sample.sample_id)
        elif _in_test(sample.as_of_date, split, policy):
            assignments["test"].append(sample.sample_id)
    return {key: tuple(value) for key, value in assignments.items()}


def reject_random_split(name: str) -> None:
    if name.strip().lower() in {"random", "shuffle", "stratified_random"}:
        raise ValueError("randomized cross-time splits are forbidden")


def _in_train(as_of_date: date, label_end: date, split: TimeSplit, policy: SplitPolicy) -> bool:
    purge_start = split.valid_start - timedelta(days=policy.purge_days)
    embargo_end = split.valid_start + timedelta(days=policy.embargo_days)
    return (
        split.train_start <= as_of_date <= split.train_end
        and label_end < purge_start
        and as_of_date < embargo_end
    )


def _in_valid(as_of_date: date, label_end: date, split: TimeSplit, policy: SplitPolicy) -> bool:
    purge_start = split.test_start - timedelta(days=policy.purge_days)
    embargo_end = split.test_start + timedelta(days=policy.embargo_days)
    return (
        split.valid_start <= as_of_date <= split.valid_end
        and as_of_date > split.train_end + timedelta(days=policy.embargo_days)
        and label_end < purge_start
        and as_of_date < embargo_end
    )


def _in_test(as_of_date: date, split: TimeSplit, policy: SplitPolicy) -> bool:
    return (
        split.test_start <= as_of_date <= split.test_end
        and as_of_date > split.valid_end + timedelta(days=policy.embargo_days)
    )
