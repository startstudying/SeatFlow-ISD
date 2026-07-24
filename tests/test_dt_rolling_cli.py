from __future__ import annotations

import json
import pickle
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.cli import _build_spilled_dt_folds, main
from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.graph.tensorize import TENSORIZATION_VERSION, TensorizedGraph


class DTRollingCliTest(unittest.TestCase):
    def test_spilled_folds_load_one_fold_contract_without_graph_rebuild(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            samples = tuple(
                TensorizedWindowSample(
                    sample_id=f"sample-{index}",
                    source_hash="a" * 64,
                    target_stock_id="000001.SZ",
                    target_node_index=0,
                    graph=TensorizedGraph(
                        data=None,
                        metadata={
                            "tensor_schema_version": TENSORIZATION_VERSION,
                            "edge_feature_schema_hash": "a" * 64,
                        },
                    ),
                    labels={1: index % 2},
                    returns={1: 0.01 if index % 2 else -0.01},
                    as_of_date=date(2024, 1, 1) + timedelta(days=index),
                    label_end_dates={1: date(2024, 1, 2) + timedelta(days=index)},
                )
                for index in range(9)
            )
            chunk_path = root / "chunk.pkl"
            with chunk_path.open("wb") as stream:
                pickle.dump(samples, stream)

            folds = _build_spilled_dt_folds(
                (chunk_path,),
                tuple(sample.as_of_date for sample in samples if sample.as_of_date),
                selection_horizon=1,
                min_train_dates=3,
                purge_days=0,
                embargo_days=0,
            )

        self.assertEqual(len(folds), 3)
        self.assertTrue(all(fold.train and fold.test for fold in folds))

    def test_spilled_folds_reject_unversioned_tensor_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample = TensorizedWindowSample(
                sample_id="legacy", source_hash="a" * 64, target_stock_id="000001.SZ",
                target_node_index=0, graph=TensorizedGraph(data=None, metadata={}),
                labels={1: 1}, returns={1: 0.01}, as_of_date=date(2024, 1, 1),
                label_end_dates={1: date(2024, 1, 2)},
            )
            chunk_path = root / "legacy.pkl"
            with chunk_path.open("wb") as stream:
                pickle.dump((sample,), stream)
            with self.assertRaisesRegex(ValueError, "obsolete or unversioned"):
                _build_spilled_dt_folds(
                    (chunk_path,),
                    tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(4)),
                    selection_horizon=1, min_train_dates=1, purge_days=0, embargo_days=0,
                )

    def test_future_blind_request_writes_negative_result_before_cli_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result_path = root / "negative.json"
            with self.assertRaises(SystemExit):
                main(
                    [
                        "dt-rolling-run",
                        "--data-root",
                        str(root / "data"),
                        "--output-root",
                        str(root / "checkpoints"),
                        "--result-file",
                        str(result_path),
                        "--as-of-start",
                        "2026-07-20",
                        "--as-of-end",
                        "2026-07-20",
                        "--observed-label-end",
                        "2026-07-20",
                    ]
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "failed_closed")
        self.assertTrue(payload["negative_result"])
        self.assertFalse(payload["model_valid"])
        self.assertEqual(payload["rows"], [])
        self.assertIn("future-blind", payload["failure_reason"])


if __name__ == "__main__":
    unittest.main()
