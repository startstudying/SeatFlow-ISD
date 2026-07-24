from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TimeSplit:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date

    def to_dict(self) -> dict[str, str]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "valid_start": self.valid_start.isoformat(),
            "valid_end": self.valid_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def validate_time_split(split: TimeSplit) -> tuple[str, ...]:
    errors: list[str] = []
    if split.train_start > split.train_end:
        errors.append("train_start must be on or before train_end")
    if split.valid_start > split.valid_end:
        errors.append("valid_start must be on or before valid_end")
    if split.test_start > split.test_end:
        errors.append("test_start must be on or before test_end")
    if split.train_end >= split.valid_start:
        errors.append("train_end must be before valid_start")
    if split.valid_end >= split.test_start:
        errors.append("valid_end must be before test_start")
    return tuple(errors)
