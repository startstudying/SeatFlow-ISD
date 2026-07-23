from __future__ import annotations

import unittest

from seatflow.evaluation.blind import assert_no_label_feedback


class BlindLabelAccessGuardTest(unittest.TestCase):
    def test_nested_labels_and_outcomes_are_rejected(self) -> None:
        for key in ("label", "labels", "future_return", "target", "outcome"):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "forbidden label keys"
            ):
                assert_no_label_feedback({"predictions": [{"metadata": {key: 1.0}}]})

    def test_prediction_only_payload_is_allowed(self) -> None:
        assert_no_label_feedback(
            {
                "prediction_date": "2026-08-01",
                "stock_id": "000001.SZ",
                "predicted_rank": 1,
                "predicted_residual_return": 0.01,
            }
        )


if __name__ == "__main__":
    unittest.main()
