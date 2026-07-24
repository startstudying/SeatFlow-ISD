from __future__ import annotations

import unittest
from datetime import date

from seatflow.graph.tensorize import TENSORIZATION_VERSION, derive_frozen_graph_schema
from seatflow.graph.types import HeteroGraphSnapshot


class TensorSchemaVersionTest(unittest.TestCase):
    def test_schema_fingerprint_is_stable_and_versioned(self) -> None:
        schema = derive_frozen_graph_schema((HeteroGraphSnapshot(date(2026, 1, 1), (), ()),))
        self.assertEqual(TENSORIZATION_VERSION, "seatflow_tensor_v3_no_id_hash")
        self.assertEqual(len(schema.fingerprint()), 64)
        self.assertEqual(schema.fingerprint(), schema.fingerprint())


if __name__ == "__main__":
    unittest.main()
