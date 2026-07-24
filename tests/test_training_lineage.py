from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.tracking.lineage import LineageRecord, LocalLineageStore
from seatflow.training.preflight import TimeSplit
from seatflow.training.runner import run_engineering_smoke


class TrainingLineageTest(unittest.TestCase):
    def test_engineering_smoke_writes_required_hashes_and_never_marks_model_valid(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            result = run_engineering_smoke(
                Path("configs/default.toml"),
                Path(temporary_directory),
                run_id="test-smoke",
            )

            self.assertEqual(result.status, "completed")
            self.assertFalse(result.model_valid)
            self.assertTrue(result.checkpoint_path.exists())
            self.assertEqual(
                set(result.hashes),
                {"data", "config", "code", "seed", "split", "blind_manifest", "checkpoint"},
            )
            lines = result.lineage_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertFalse(record["model_valid"])
            self.assertEqual(record["run_type"], "engineering_smoke")
            self.assertIn("sample_only", record["tags"])

    def test_lineage_is_append_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            first = run_engineering_smoke(
                Path("configs/default.toml"),
                Path(temporary_directory),
                run_id="test-smoke-a",
            )
            second = run_engineering_smoke(
                Path("configs/default.toml"),
                Path(temporary_directory),
                run_id="test-smoke-b",
            )

            self.assertEqual(first.lineage_path, second.lineage_path)
            lines = second.lineage_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_id"], "test-smoke-a")
            self.assertEqual(json.loads(lines[1])["run_id"], "test-smoke-b")

    def test_engineering_smoke_fails_closed_for_non_sample_provider(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "non_sample.toml"
            config_path.write_text(
                Path("configs/default.toml")
                .read_text(encoding="utf-8")
                .replace('provider = "sample"', 'provider = "tushare"'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "provider='sample'"):
                run_engineering_smoke(config_path, root / "out", run_id="blocked")

    def test_local_lineage_rejects_model_valid_without_real_evidence(self) -> None:
        split = TimeSplit(
            train_start=date(2021, 1, 1),
            train_end=date(2021, 6, 30),
            valid_start=date(2021, 7, 1),
            valid_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 6, 30),
        )
        with TemporaryDirectory() as temporary_directory:
            store = LocalLineageStore(Path(temporary_directory))
            with self.assertRaisesRegex(ValueError, "data_scope:real"):
                store.append(
                    LineageRecord(
                        run_id="bad",
                        run_type="engineering_smoke",
                        status="completed",
                        model_valid=True,
                        provider="sample",
                        seed=42,
                        split=split,
                        hashes={
                            "data": "1",
                            "config": "2",
                            "code": "3",
                            "seed": "4",
                            "split": "5",
                            "blind_manifest": "6",
                            "checkpoint": "7",
                        },
                        metrics={},
                        tags=(),
                        created_at="2026-01-01T00:00:00+08:00",
                    )
                )

    def test_local_lineage_allows_model_valid_with_real_protocol_evidence(self) -> None:
        split = TimeSplit(
            train_start=date(2021, 1, 1),
            train_end=date(2021, 6, 30),
            valid_start=date(2021, 7, 1),
            valid_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 6, 30),
        )
        with TemporaryDirectory() as temporary_directory:
            path = LocalLineageStore(Path(temporary_directory)).append(
                LineageRecord(
                    run_id="real-validated",
                    run_type="train",
                    status="completed",
                    model_valid=True,
                    provider="tushare",
                    seed=42,
                    split=split,
                    hashes={
                        "data": "1",
                        "config": "2",
                        "code": "3",
                        "seed": "4",
                        "split": "5",
                        "blind_manifest": "6",
                        "checkpoint": "7",
                        "protocol": "8",
                        "evaluation": "9",
                    },
                    metrics={"auc": 0.6},
                    tags=("data_scope:real", "walk_forward"),
                    created_at="2026-01-01T00:00:00+08:00",
                )
            )

            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(record["model_valid"])
            self.assertEqual(record["provider"], "tushare")


if __name__ == "__main__":
    unittest.main()
