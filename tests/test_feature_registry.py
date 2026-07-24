from __future__ import annotations

import unittest

from seatflow.feature_registry import FeatureKind, infer_feature_kind


class FeatureRegistryTest(unittest.TestCase):
    def test_entity_identifiers_are_not_continuous_features(self) -> None:
        self.assertEqual(infer_feature_kind("stock_id_hash"), FeatureKind.ENTITY_ID)
        self.assertEqual(infer_feature_kind("seat_id"), FeatureKind.ENTITY_ID)

    def test_date_hashes_are_rejected_as_numeric_identity_fields(self) -> None:
        self.assertEqual(infer_feature_kind("window_start_hash"), FeatureKind.ENTITY_ID)


if __name__ == "__main__":
    unittest.main()
