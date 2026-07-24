from __future__ import annotations

import unittest

from seatflow.evaluation.ensemble import equal_weight_seed_scores


class SeedEnsembleEqualWeightTest(unittest.TestCase):
    def test_all_five_seeds_receive_exactly_equal_weight(self) -> None:
        predictions = {
            seed: {"000001.SZ": float(index), "000002.SZ": float(index + 5)}
            for index, seed in enumerate((2026, 2027, 2028, 2029, 2030), start=1)
        }
        ensemble = equal_weight_seed_scores(predictions)

        self.assertEqual(ensemble["000001.SZ"], 3.0)
        self.assertEqual(ensemble["000002.SZ"], 8.0)

    def test_missing_or_failed_seed_cannot_be_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "every frozen seed"):
            equal_weight_seed_scores(
                {seed: {"000001.SZ": 1.0} for seed in (2026, 2027, 2028, 2029)}
            )


if __name__ == "__main__":
    unittest.main()
