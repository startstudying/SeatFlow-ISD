from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.evaluation.blind import (
    assert_no_label_feedback,
    freeze_blind_manifest,
    freeze_evaluation_manifest,
    verify_blind_manifest,
    verify_evaluation_manifest,
)


class BlindManifestTest(unittest.TestCase):
    def test_freeze_and_verify_manifest_hashes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature = root / "features.jsonl"
            label = root / "labels.jsonl"
            feature.write_text('{"example_id":"a","score":0.1}\n', encoding="utf-8")
            label.write_text('{"example_id":"a","future_return":0.2}\n', encoding="utf-8")

            frozen = freeze_blind_manifest(
                root / "blind.json",
                feature_paths=(feature,),
                label_paths=(label,),
                metadata={"split": "blind"},
            )
            verified = verify_blind_manifest(frozen.path)

            self.assertTrue(verified["verified"])
            self.assertEqual(verified["feature_file_count"], 1)
            self.assertEqual(frozen.label_file_count, 1)

    def test_verify_manifest_rejects_tampered_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature = root / "features.jsonl"
            label = root / "labels.jsonl"
            feature.write_text('{"example_id":"a"}\n', encoding="utf-8")
            label.write_text('{"example_id":"a","label":1}\n', encoding="utf-8")
            frozen = freeze_blind_manifest(
                root / "blind.json",
                feature_paths=(feature,),
                label_paths=(label,),
            )

            feature.write_text('{"example_id":"a","label":1}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_blind_manifest(frozen.path)

    def test_forbids_label_feedback_payload_keys(self) -> None:
        assert_no_label_feedback({"examples": [{"score": 0.1}]})
        with self.assertRaisesRegex(ValueError, "forbidden label keys"):
            assert_no_label_feedback({"examples": [{"future_return": 0.2}]})

    def test_contaminated_oos_manifest_is_not_accepted_as_future_blind(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature = root / "features.jsonl"
            label = root / "labels.jsonl"
            feature.write_text('{"example_id":"a","score":0.1}\n', encoding="utf-8")
            label.write_text('{"example_id":"a","label":1}\n', encoding="utf-8")
            frozen = freeze_evaluation_manifest(
                root / "contaminated.json",
                feature_paths=(feature,),
                label_paths=(label,),
                split_status="contaminated_oos",
            )

            verified = verify_evaluation_manifest(frozen.path)

            self.assertEqual(verified["split_status"], "contaminated_oos")
            with self.assertRaisesRegex(ValueError, "future_blind"):
                verify_blind_manifest(frozen.path)


if __name__ == "__main__":
    unittest.main()
