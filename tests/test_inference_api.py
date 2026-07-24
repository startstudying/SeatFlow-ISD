import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from seatflow.api import create_app
from seatflow.data.store import SnapshotStore

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None  # type: ignore[assignment, misc]


@unittest.skipIf(TestClient is None, "FastAPI test client is not installed")
class InferenceApiTest(unittest.TestCase):
    def test_prediction_fails_closed_when_checkpoint_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = TestClient(create_app(temporary_directory, "sample"))

            response = client.post(
                "/api/v1/providers/sample/models/missing-model/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

            self.assertEqual(response.status_code, 404)
            payload = response.json()
            self.assertEqual(payload["model_validity_status"], "unavailable")
            self.assertEqual(payload["data_scope"], "unknown")
            self.assertIsNone(payload["lineage_hash"])
            self.assertIn("checkpoint_missing", payload["risk_flags"])

    def test_prediction_fails_closed_when_lineage_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_model_fixture(root, model_id="seat-hgt-v1", lineage_hash="lineage-real")
            client = TestClient(create_app(root, "sample"))

            response = client.post(
                "/api/v1/providers/sample/models/seat-hgt-v1/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

            self.assertEqual(response.status_code, 409)
            payload = response.json()
            self.assertEqual(payload["model_validity_status"], "valid")
            self.assertEqual(payload["data_scope"], "real")
            self.assertEqual(payload["lineage_hash"], "lineage-real")
            self.assertIn("lineage_missing", payload["risk_flags"])

    def test_prediction_fails_closed_for_non_real_graph_snapshot(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_model_fixture(root, model_id="seat-hgt-v1", lineage_hash="lineage-real")
            _write_contract_artifact(root, "artifacts/lineage/lineage-real.json")
            _write_graph_fixture(
                root, metadata={"data_scope": "sample", "seat_detail_scope": "synthetic"}
            )
            client = TestClient(create_app(root, "sample"))

            response = client.post(
                "/api/v1/providers/sample/models/seat-hgt-v1/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

            self.assertEqual(response.status_code, 409)
            payload = response.json()
            self.assertEqual(payload["data_scope"], "sample")
            self.assertEqual(payload["lineage_hash"], "lineage-real")
            self.assertIn("non_real_graph_data_scope", payload["risk_flags"])

    def test_model_run_backtest_and_prediction_return_persisted_contract_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_model_fixture(root, model_id="seat-hgt-v1", lineage_hash="lineage-real")
            _write_contract_artifact(root, "artifacts/lineage/lineage-real.json")
            _write_contract_artifact(root, "artifacts/runs/run-20260115.json")
            _write_contract_artifact(root, "artifacts/backtests/run-20260115.json")
            _write_graph_fixture(
                root,
                metadata={"data_scope": "real", "seat_detail_scope": "complete_seat"},
            )
            _write_contract_artifact(
                root,
                "artifacts/predictions/seat-hgt-v1/sample/2026-01-15/600000.SH.json",
                extra={
                    "model_id": "seat-hgt-v1",
                    "provider": "sample",
                    "trade_date": "2026-01-15",
                    "target_stock_id": "600000.SH",
                    "rise_probability": 0.61,
                    "expected_returns": {"1": 0.012},
                },
            )
            client = TestClient(create_app(root, "sample"))

            model_response = client.get("/api/v1/models/seat-hgt-v1")
            run_response = client.get("/api/v1/runs/run-20260115")
            backtest_response = client.get("/api/v1/backtests/run-20260115")
            predict_response = client.post(
                "/api/v1/providers/sample/models/seat-hgt-v1/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

            for response in (model_response, run_response, backtest_response, predict_response):
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["model_validity_status"], "valid")
                self.assertEqual(payload["data_scope"], "real")
                self.assertEqual(payload["lineage_hash"], "lineage-real")
                self.assertEqual(payload["risk_flags"], [])

            prediction_payload = predict_response.json()
            self.assertEqual(prediction_payload["prediction"]["rise_probability"], 0.61)
            self.assertEqual(prediction_payload["graph_manifest"]["metadata"]["data_scope"], "real")

    def test_prediction_rejects_artifact_with_mismatched_lineage_binding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_model_fixture(root, model_id="seat-hgt-v1", lineage_hash="lineage-real")
            _write_contract_artifact(root, "artifacts/lineage/lineage-real.json")
            _write_graph_fixture(
                root,
                metadata={"data_scope": "real", "seat_detail_scope": "complete_seat"},
            )
            _write_contract_artifact(
                root,
                "artifacts/predictions/seat-hgt-v1/sample/2026-01-15/600000.SH.json",
                extra={
                    "model_id": "seat-hgt-v1",
                    "provider": "sample",
                    "trade_date": "2026-01-15",
                    "target_stock_id": "600000.SH",
                    "lineage_hash": "wrong-lineage",
                },
            )
            client = TestClient(create_app(root, "sample"))

            response = client.post(
                "/api/v1/providers/sample/models/seat-hgt-v1/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("prediction_lineage_hash_mismatch", response.json()["risk_flags"])

    def test_prediction_rejects_artifact_with_mismatched_scope_binding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_model_fixture(root, model_id="seat-hgt-v1", lineage_hash="lineage-real")
            _write_contract_artifact(root, "artifacts/lineage/lineage-real.json")
            _write_graph_fixture(
                root,
                metadata={"data_scope": "real", "seat_detail_scope": "complete_seat"},
            )
            _write_contract_artifact(
                root,
                "artifacts/predictions/seat-hgt-v1/sample/2026-01-15/600000.SH.json",
                extra={
                    "model_id": "seat-hgt-v1",
                    "provider": "sample",
                    "trade_date": "2026-01-15",
                    "target_stock_id": "600000.SH",
                    "seat_detail_scope": "institutional_only",
                },
            )
            client = TestClient(create_app(root, "sample"))

            response = client.post(
                "/api/v1/providers/sample/models/seat-hgt-v1/predict",
                json={"trade_date": "2026-01-15", "target_stock_id": "600000.SH"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("prediction_seat_detail_scope_mismatch", response.json()["risk_flags"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_model_fixture(root: Path, *, model_id: str, lineage_hash: str) -> None:
    model_dir = root / "checkpoints" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.ckpt").write_text("fixture checkpoint\n", encoding="utf-8")
    _write_json(
        model_dir / "metadata.json",
        {
            "model_id": model_id,
            "run_id": "run-20260115",
            "checkpoint_file": "model.ckpt",
            "model_validity_status": "valid",
            "data_scope": "real",
            "seat_detail_scope": "complete_seat",
            "lineage_hash": lineage_hash,
            "risk_flags": [],
        },
    )


def _write_contract_artifact(
    root: Path, relative_path: str, extra: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {
        "model_validity_status": "valid",
        "data_scope": "real",
        "seat_detail_scope": "complete_seat",
        "lineage_hash": "lineage-real",
        "risk_flags": [],
    }
    payload.update(extra or {})
    _write_json(root / relative_path, payload)


def _write_graph_fixture(root: Path, metadata: dict[str, Any]) -> None:
    graph = {
        "as_of_date": "2026-01-15",
        "nodes": [{"node_type": "stock", "node_id": "600000.SH", "features": {}}],
        "edges": [],
    }
    SnapshotStore(root / "graphs").write(
        "sample",
        "hetero_graph",
        date(2026, 1, 15),
        [graph],
        metadata=metadata,
    )


if __name__ == "__main__":
    unittest.main()
