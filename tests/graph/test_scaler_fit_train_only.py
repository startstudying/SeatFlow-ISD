from __future__ import annotations

import unittest

from seatflow.feature_transforms import apply_feature_transform, fit_feature_transform


class ScalerFitTrainOnlyTest(unittest.TestCase):
    def test_validation_outlier_cannot_change_robust_state(self) -> None:
        state = fit_feature_transform(
            ({"amount": 1.0}, {"amount": 2.0}, {"amount": 3.0}),
            ("amount",),
            sample_ids=("a", "b", "c"),
            manifest_hash="a" * 64,
            pretransformed_features=("amount",),
        )
        before = state.fingerprint()
        output = apply_feature_transform(({"amount": 1_000_000.0},), state)

        self.assertEqual(state.fingerprint(), before)
        self.assertEqual(state.fitted_sample_count, 3)
        self.assertLessEqual(abs(output[0][0]), 8.0)


if __name__ == "__main__":
    unittest.main()
