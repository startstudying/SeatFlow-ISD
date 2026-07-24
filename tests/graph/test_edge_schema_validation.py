from __future__ import annotations

import unittest
from math import log1p

from seatflow.graph.edge_schema import (
    EdgeFeatureType,
    classify_edge_field,
    transform_edge_value,
    validate_edge_attributes,
)


class EdgeSchemaValidationTest(unittest.TestCase):
    def test_identifier_and_absolute_timestamp_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "identifier"):
            validate_edge_attributes({"stock_id_hash": 1.0})
        with self.assertRaisesRegex(ValueError, "relative time"):
            validate_edge_attributes({"event_timestamp": 20260101.0})

    def test_ratio_and_count_contracts_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "ratio"):
            validate_edge_attributes({"amount_share_of_stock": 1.01})
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            validate_edge_attributes({"trade_count": 1.5})

        self.assertEqual(classify_edge_field("trade_count").kind, EdgeFeatureType.COUNT)
        self.assertAlmostEqual(transform_edge_value("trade_count", 4.0), log1p(4.0))

    def test_relative_time_is_allowed(self) -> None:
        validate_edge_attributes({"last_trade_age": 2.0})
        self.assertTrue(classify_edge_field("last_trade_age").relative_time)


if __name__ == "__main__":
    unittest.main()
