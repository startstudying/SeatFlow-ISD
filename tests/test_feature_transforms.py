from __future__ import annotations

import unittest

from seatflow.feature_transforms import (
    FeatureTransformError,
    FeatureTransformState,
    apply_feature_transform,
    feature_transform_diagnostics,
    fit_feature_transform,
)

_MANIFEST_HASH = "a" * 64


class FeatureTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.order = ("net_amount", "momentum_5d", "turnover_ratio", "is_active")
        self.rows = (
            {"net_amount": -1000.0, "momentum_5d": -0.10, "turnover_ratio": 0.1, "is_active": 1.0},
            {"net_amount": 0.0, "momentum_5d": 0.00, "turnover_ratio": 0.2, "is_active": 0.0},
            {"net_amount": 1000.0, "momentum_5d": 0.10, "turnover_ratio": 0.3, "is_active": 1.0},
        )

    def test_fit_apply_round_trip_is_deterministic_and_serializable(self) -> None:
        state = self._fit()
        restored = FeatureTransformState.from_dict(state.to_dict())
        self.assertEqual(state.fingerprint(), restored.fingerprint())
        self.assertEqual(
            apply_feature_transform(self.rows, state), apply_feature_transform(self.rows, restored)
        )

    def test_amount_is_signed_log_and_test_outlier_cannot_influence_fit(self) -> None:
        state = self._fit()
        test_row = {
            "net_amount": 1e100,
            "momentum_5d": 1e100,
            "turnover_ratio": 0.5,
            "is_active": 1.0,
        }
        output = apply_feature_transform((test_row,), state)[0]
        self.assertTrue(all(abs(value) < 100.0 for value in output))
        self.assertEqual(state.statistics["net_amount"].transform, "signed_log_standardize")
        self.assertEqual(state.statistics["turnover_ratio"].transform, "identity")

    def test_schema_entity_nonfinite_and_order_mismatches_fail_closed(self) -> None:
        with self.assertRaisesRegex(FeatureTransformError, "non-continuous"):
            fit_feature_transform(
                ({"stock_id_hash": 1.0},),
                ("stock_id_hash",),
                sample_ids=("a",),
                manifest_hash=_MANIFEST_HASH,
            )
        state = self._fit()
        with self.assertRaisesRegex(FeatureTransformError, "feature order"):
            apply_feature_transform(self.rows, state, feature_order=tuple(reversed(self.order)))
        bad = dict(self.rows[0])
        bad["net_amount"] = float("nan")
        with self.assertRaisesRegex(FeatureTransformError, "non-finite"):
            apply_feature_transform((bad,), state)

    def test_zero_variance_features_remain_finite_without_refitting(self) -> None:
        rows = ({"amount": 7.0}, {"amount": 7.0})
        state = fit_feature_transform(
            rows, ("amount",), sample_ids=("a", "b"), manifest_hash=_MANIFEST_HASH
        )
        self.assertEqual(apply_feature_transform(rows, state), ((0.0,), (0.0,)))
        self.assertEqual(apply_feature_transform(({"amount": 8.0},), state), ((0.0,),))
        diagnostics = feature_transform_diagnostics(rows, state)
        self.assertEqual(diagnostics["distributions"]["amount"]["finite_count"], 2)
        self.assertEqual(diagnostics["distributions"]["amount"]["missing_count"], 0)
        self.assertIn("p99", diagnostics["distributions"]["amount"])
        self.assertEqual(len(str(diagnostics["diagnostics_hash"])), 64)

    def _fit(self):  # type: ignore[no-untyped-def]
        return fit_feature_transform(
            self.rows, self.order, sample_ids=("a", "b", "c"), manifest_hash=_MANIFEST_HASH
        )


if __name__ == "__main__":
    unittest.main()
