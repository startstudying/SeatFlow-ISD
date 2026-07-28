"""Week-5 fix: verify seat amount log1p normalization prevents E3 Train=NaN.

Raw Dragon-Tiger seat amounts reach ~1e10 yuan. Without data-side
normalization the E3 seat branch produces NaN even with LayerNorm after
the projection. This test pins the sign-preserving log1p contract.
"""

from __future__ import annotations

import unittest
from math import isfinite, log1p

from seatflow.r4_features.seat_aggregates import _signed_log1p


class SignedLog1pTest(unittest.TestCase):
    def test_zero_stays_zero(self) -> None:
        self.assertEqual(_signed_log1p(0.0), 0.0)

    def test_positive_value_uses_log1p(self) -> None:
        self.assertAlmostEqual(_signed_log1p(400.0), log1p(400.0))

    def test_negative_value_preserves_sign(self) -> None:
        # -200 -> -(log1p(200))
        self.assertAlmostEqual(_signed_log1p(-200.0), -log1p(200.0))
        self.assertLess(_signed_log1p(-200.0), 0.0)

    def test_large_monetary_amount_does_not_explode(self) -> None:
        # ~1e10 yuan (a single seat's net buy on a hot Dragon-Tiger day).
        # Raw value would overflow fp32 attention logits; log1p maps it to ~23.
        normalized = _signed_log1p(1e10)
        self.assertTrue(isfinite(normalized))
        self.assertLess(normalized, 30.0)
        self.assertGreater(normalized, 20.0)

    def test_ordering_preserved(self) -> None:
        # log1p is monotonic, so ordering survives normalization — important
        # for the ranking head.
        self.assertLess(_signed_log1p(100.0), _signed_log1p(1e8))
        self.assertLess(_signed_log1p(1e8), _signed_log1p(1e10))

    def test_nan_collapses_to_zero(self) -> None:
        self.assertEqual(_signed_log1p(float("nan")), 0.0)

    def test_inf_collapses_to_zero(self) -> None:
        self.assertEqual(_signed_log1p(float("inf")), 0.0)
        self.assertEqual(_signed_log1p(float("-inf")), 0.0)


if __name__ == "__main__":
    unittest.main()
