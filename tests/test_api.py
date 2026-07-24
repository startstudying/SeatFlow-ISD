import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.api import create_app
from seatflow.data.store import SnapshotStore

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None  # type: ignore[assignment, misc]


@unittest.skipIf(TestClient is None, "FastAPI test client is not installed")
class ApiTest(unittest.TestCase):
    def test_snapshot_summary_and_graph_are_read_from_snapshot_store(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph = {
                "as_of_date": "2026-01-15",
                "nodes": [{"node_type": "stock", "node_id": "600000.SH", "features": {}}],
                "edges": [],
            }
            SnapshotStore(root / "processed").write(
                "sample",
                "daily_bar",
                date(2026, 1, 15),
                [{"stock_id": "600000.SH", "trade_date": "2026-01-15"}],
            )
            SnapshotStore(root / "graphs").write(
                "sample",
                "hetero_graph",
                date(2026, 1, 15),
                [graph],
                metadata={"node_count": 1, "edge_count": 0},
            )

            client = TestClient(create_app(root, "sample"))

            summary_response = client.get("/api/v1/providers/sample/snapshot-summary")
            self.assertEqual(summary_response.status_code, 200)
            summary = summary_response.json()
            self.assertEqual(summary["provider"], "sample")
            self.assertEqual(summary["snapshot_count"], 2)
            self.assertEqual(
                {(item["store"], item["dataset"]) for item in summary["datasets"]},
                {("processed", "daily_bar"), ("graphs", "hetero_graph")},
            )

            graph_response = client.get("/api/v1/providers/sample/graph/2026-01-15")
            self.assertEqual(graph_response.status_code, 200)
            graph_payload = graph_response.json()
            self.assertEqual(graph_payload["manifest"]["row_count"], 1)
            self.assertEqual(graph_payload["graph"]["nodes"][0]["node_id"], "600000.SH")

    def test_missing_provider_or_graph_returns_404_without_synthetic_data(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = TestClient(create_app(temporary_directory, "sample"))

            health_response = client.get("/health")
            self.assertEqual(health_response.status_code, 200)
            self.assertEqual(health_response.json()["status"], "ok")

            summary_response = client.get("/api/v1/providers/missing/snapshot-summary")
            self.assertEqual(summary_response.status_code, 404)

            graph_response = client.get("/api/v1/providers/sample/graph/2026-01-15")
            self.assertEqual(graph_response.status_code, 404)

            status_response = client.get("/api/v1/audit/status")
            self.assertEqual(status_response.status_code, 200)
            self.assertFalse(status_response.json()["data_ready"])
            self.assertEqual(status_response.json()["scope"], "none")
