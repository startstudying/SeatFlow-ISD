from __future__ import annotations

import unittest

from seatflow.r4_features import (
    B1_FEATURE_COLUMN_PROTOCOL,
    SEAT_AGGREGATE_MODEL_FEATURES,
    SeatAggregateFeatureError,
    TrainOnlyStandardScaler,
)


class R4NoIdentityHashTest(unittest.TestCase):
    def test_b1_protocol_contains_no_numeric_identity_or_hash_columns(self) -> None:
        for name in (*B1_FEATURE_COLUMN_PROTOCOL, *SEAT_AGGREGATE_MODEL_FEATURES):
            normalized = name.lower()
            self.assertNotIn("hash", normalized)
            self.assertNotIn("_id", normalized)
            self.assertNotIn("id_", normalized)

    def test_scaler_rejects_identity_hash_continuous_features(self) -> None:
        for feature_name in ("stock_id_hash", "seat_id_hash", "identity_code"):
            with self.subTest(feature_name=feature_name), self.assertRaisesRegex(
                SeatAggregateFeatureError, "identity/hash"
            ):
                TrainOnlyStandardScaler((feature_name,))


if __name__ == "__main__":
    unittest.main()
