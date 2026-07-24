import unittest
from datetime import date

from seatflow.training.preflight import TimeSplit, validate_time_split


class TrainingPreflightTest(unittest.TestCase):
    def test_rejects_overlapping_periods(self) -> None:
        split = TimeSplit(
            train_start=date(2021, 1, 1),
            train_end=date(2024, 7, 1),
            valid_start=date(2024, 7, 1),
            valid_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            test_end=date(2025, 6, 30),
        )

        errors = validate_time_split(split)

        self.assertIn("train_end must be before valid_start", errors)
