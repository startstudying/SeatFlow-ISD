from __future__ import annotations

import unittest

from seatflow.evaluation.oracle_audit import score_direction_consistency


class ScoreDirectionConsistencyTest(unittest.TestCase):
    def test_equal_score_and_direction_probability_is_consistent(self) -> None:
        rows = [
            {"score": 0.8, "direction_probability": 0.8},
            {"score": 0.3, "direction_probability": 0.3},
        ]
        diag = score_direction_consistency(rows)

        self.assertEqual(diag.row_count, 2)
        self.assertAlmostEqual(diag.mean_abs_delta, 0.0)
        self.assertTrue(diag.max_delta_below_tolerance)

    def test_non_matching_scores_are_reported(self) -> None:
        rows = [
            {"score": 0.8, "direction_probability": 0.1},
            {"score": 0.3, "direction_probability": 0.7},
        ]
        diag = score_direction_consistency(rows, tolerance=0.01)

        self.assertFalse(diag.max_delta_below_tolerance)
        self.assertAlmostEqual(diag.max_abs_delta, 0.7)


if __name__ == "__main__":
    unittest.main()
