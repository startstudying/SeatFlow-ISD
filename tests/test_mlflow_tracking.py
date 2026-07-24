from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from seatflow.tracking.lineage import LineageRecord, export_run_package, record_training_run
from seatflow.training.preflight import TimeSplit


class _FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="mlflow-run-1")

    def __enter__(self) -> _FakeRun:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.logged_dicts: list[tuple[dict[str, object], str]] = []
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, run_name: str) -> _FakeRun:
        self.run_name = run_name
        return _FakeRun()

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_dict(self, payload: dict[str, object], artifact_file: str) -> None:
        self.logged_dicts.append((payload, artifact_file))


class MlflowTrackingTest(unittest.TestCase):
    def test_enabled_mlflow_missing_falls_back_to_append_only_local_lineage(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "mlflow.toml"
            config_path.write_text(
                '[mlflow]\nenabled = true\ntracking_uri = "file:./mlruns"\n',
                encoding="utf-8",
            )
            with patch(
                "seatflow.tracking.lineage.import_module",
                side_effect=ModuleNotFoundError("mlflow"),
            ):
                result = record_training_run(_record("fallback"), root, config_path)

            self.assertEqual(result.backend, "local-jsonl")
            self.assertEqual(result.fallback_reason, "mlflow_not_installed")
            lines = result.lineage_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["run_id"], "fallback")

    def test_enabled_mlflow_mirrors_run_while_keeping_local_lineage(self) -> None:
        fake = _FakeMlflow()
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "mlflow.toml"
            config_path.write_text(
                '[mlflow]\nenabled = true\ntracking_uri = "file:./mlruns"\n'
                'experiment_name = "SeatFlowTest"\n',
                encoding="utf-8",
            )
            with patch("seatflow.tracking.lineage.import_module", return_value=fake):
                result = record_training_run(_record("mirrored"), root, config_path)

            self.assertEqual(result.backend, "mlflow+local-jsonl")
            self.assertEqual(result.mlflow_run_id, "mlflow-run-1")
            self.assertEqual(fake.experiment_name, "SeatFlowTest")
            self.assertEqual(fake.params["run_id"], "mirrored")
            self.assertEqual(fake.metrics["auc"], 0.51)
            self.assertEqual(fake.logged_dicts[0][1], "lineage/run.json")
            mirrored_payload = fake.logged_dicts[0][0]
            hashes = mirrored_payload["hashes"]
            self.assertIsInstance(hashes, dict)
            self.assertEqual(cast(dict[str, Any], hashes)["api_token"], "<redacted>")
            self.assertNotIn("secret", json.dumps(mirrored_payload))
            self.assertNotIn("private-value", fake.tags["seatflow.tags"])
            self.assertNotIn("api-token-value", fake.tags["seatflow.tags"])
            self.assertNotIn("access-token-value", fake.tags["seatflow.tags"])
            self.assertIn("token=<redacted>", fake.tags["seatflow.tags"])
            self.assertIn("api_token=<redacted>", fake.tags["seatflow.tags"])
            self.assertIn("access_token=<redacted>", fake.tags["seatflow.tags"])
            self.assertTrue(result.lineage_path.exists())

    def test_export_run_package_redacts_secret_keys_and_refuses_raw_data(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = record_training_run(_record("exported"), root, root / "missing.toml")
            lineage_record = json.loads(result.lineage_path.read_text(encoding="utf-8"))
            self.assertEqual(lineage_record["hashes"]["api_token"], "<redacted>")
            lineage_content = result.lineage_path.read_text(encoding="utf-8")
            self.assertNotIn("secret", lineage_content)
            self.assertNotIn("private-value", lineage_content)
            self.assertNotIn("bearer-token-value", lineage_content)
            self.assertNotIn("json-token-value", lineage_content)
            self.assertNotIn("api-token-value", lineage_content)
            self.assertNotIn("access-token-value", lineage_content)
            self.assertNotIn("access-key-value", lineage_content)
            self.assertIn("token=<redacted>", lineage_content)
            self.assertIn("api_token=<redacted>", lineage_content)
            self.assertIn("access_token=<redacted>", lineage_content)
            self.assertIn("access_key=<redacted>", lineage_content)
            self.assertIn("Bearer <redacted>", lineage_content)
            metrics = cast(dict[str, Any], lineage_record["metrics"])
            self.assertIn('"token":"<redacted>"', metrics["audit_note"])
            package = export_run_package(result.lineage_path, "exported", root / "packages")
            run_json = json.loads((package.path / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_json["hashes"]["api_token"], "<redacted>")
            self.assertFalse(
                json.loads((package.path / "manifest.json").read_text(encoding="utf-8"))[
                    "contains_tokens"
                ]
            )

            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw_file = raw_dir / "source.csv"
            raw_file.write_text("unredacted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw or secret-bearing"):
                export_run_package(
                    result.lineage_path,
                    "exported",
                    root / "blocked-package",
                    include_artifacts=(raw_file,),
                )


def _record(run_id: str) -> LineageRecord:
    return LineageRecord(
        run_id=run_id,
        run_type="engineering_smoke",
        status="completed",
        model_valid=False,
        provider="sample",
        seed=42,
        split=TimeSplit(
            train_start=date(2021, 1, 1),
            train_end=date(2021, 6, 30),
            valid_start=date(2021, 7, 1),
            valid_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 6, 30),
        ),
        hashes={
            "data": "1",
            "config": "2",
            "code": "3",
            "seed": "4",
            "split": "5",
            "blind_manifest": "6",
            "checkpoint": "7",
            "api_token": "secret",
        },
        metrics={
            "auc": 0.51,
            "model_valid": False,
            "audit_note": (
                'Authorization: Bearer bearer-token-value {"token":"json-token-value"} '
                "access_key=access-key-value"
            ),
        },
        tags=(
            "sample_only",
            "not_model_valid",
            "token=private-value",
            "api_token=api-token-value",
            "access_token=access-token-value",
        ),
        created_at="2026-01-01T00:00:00+08:00",
    )


if __name__ == "__main__":
    unittest.main()
