from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

_MANIFEST_PATH = Path("artifacts/r2_frozen_manifest.json")


class R2FrozenManifestTest(unittest.TestCase):
    def test_manifest_hashes_match_immutable_artifacts(self) -> None:
        if not _MANIFEST_PATH.exists():
            self.skipTest("r2 frozen manifest artifact not present in this checkout")
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertFalse(manifest["model_valid"])
        self.assertFalse(manifest["complete_15_run_evidence"])
        self.assertEqual(manifest["protocol"]["missing_development_run_count"], 15)
        for item in manifest["artifacts"]:
            path = Path(item["path"])
            if not path.exists():
                self.skipTest(f"referenced artifact not present: {path}")
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
