import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.datasets.v2_labels import load_v2_label_artifact


class V2LabelArtifactTest(unittest.TestCase):
    def test_loads_the_real_source_hashed_candidate_artifact(self) -> None:
        artifact = load_v2_label_artifact(
            Path("artifacts/diagnostics/R0_V2_CANDIDATE_DATASET/result.json")
        )

        self.assertEqual(len(artifact.labels), 3)
        key = (artifact.labels[0].as_of_date, "000001.SZ")
        labels = artifact.labels_by_key()[key]
        self.assertEqual([label.horizon for label in labels], [1, 3, 5])

    def test_rejects_an_artifact_with_a_tampered_payload(self) -> None:
        source = Path("artifacts/diagnostics/R0_V2_CANDIDATE_DATASET/result.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["labels"][0]["direction_label"] = 0
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                load_v2_label_artifact(path)
