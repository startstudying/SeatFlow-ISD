"""Tests for the FastAPI application: health check, pagination, error handling."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seatflow.api.app import create_app
from seatflow.api.routes.state import SYNTHETIC_NON_FORMAL


@pytest.fixture()
def client():
    """Return a client with an explicitly non-formal synthetic fixture."""
    app = create_app(reset_store=True, fixture_mode=SYNTHETIC_NON_FORMAL)
    with TestClient(app) as c:
        yield c


# ── Health check ────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_health_reports_process_up_and_research_blocked(self, client):
        resp = client.get("/research/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "DEGRADED"
        assert body["components"]["process"] == {"status": "UP"}
        assert body["components"]["researchCapability"] == {
            "status": "BLOCKED",
            "code": "PIPELINE_ADAPTER_UNAVAILABLE",
            "remediation": "Configure a real ResearchPipelineAdapter before running research tasks.",
        }
        assert "timestamp" in body

    def test_health_reports_research_up_when_adapter_is_configured(self):
        class ConfiguredAdapter:
            async def execute(self, *, task_kind, spec, report_progress):
                return {
                    "artifact_lineage": {"manifest_sha256": "a" * 64},
                }

        app = create_app(reset_store=True, pipeline_adapter=ConfiguredAdapter())
        with TestClient(app) as configured_client:
            body = configured_client.get("/research/v1/health").json()

        assert body["status"] == "UP"
        assert body["components"]["process"] == {"status": "UP"}
        assert body["components"]["researchCapability"]["status"] == "UP"

    def test_health_has_utc_timestamp(self, client):
        resp = client.get("/research/v1/health")
        ts = resp.json()["timestamp"]
        # ISO 8601 with timezone offset (ends with +00:00 or Z).
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_health_content_type_json(self, client):
        resp = client.get("/research/v1/health")
        assert "application/json" in resp.headers["content-type"]


# ── Pagination ──────────────────────────────────────────────────────────


class TestPagination:
    def test_default_pagination(self, client):
        resp = client.get("/research/v1/projects")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "meta" in body
        meta = body["meta"]
        assert meta["page"] == 1
        assert meta["page_size"] == 50
        assert meta["total"] >= 0

    def test_custom_page_size(self, client):
        resp = client.get("/research/v1/projects?page=1&page_size=1")
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["meta"]["page_size"] == 1
        assert body["meta"]["total_pages"] >= 1
        assert body["meta"]["hasNext"] is True

    def test_second_page(self, client):
        resp = client.get("/research/v1/projects?page=2&page_size=1")
        body = resp.json()
        assert body["meta"]["page"] == 2
        assert body["meta"]["hasPrev"] is True

    def test_page_size_bounds_reject_zero(self, client):
        resp = client.get("/research/v1/projects?page_size=0")
        assert resp.status_code == 422

    def test_page_size_bounds_reject_large(self, client):
        resp = client.get("/research/v1/projects?page_size=501")
        assert resp.status_code == 422

    def test_pagination_meta_total_matches_items(self, client):
        resp = client.get("/research/v1/projects?page_size=50")
        body = resp.json()
        assert body["meta"]["total"] == len(body["items"])

    def test_empty_page_returns_empty_items(self, client):
        resp = client.get("/research/v1/projects?page=999&page_size=10")
        body = resp.json()
        assert body["items"] == []
        assert body["meta"]["hasNext"] is False
        assert body["meta"]["hasPrev"] is True


class TestDefaultState:
    def test_production_factory_starts_with_empty_research_entities(self):
        app = create_app(reset_store=True)
        endpoints = (
            "projects",
            "data-sources",
            "datasets",
            "experiments",
            "runs",
            "models",
            "predictions",
            "backtests",
        )

        with TestClient(app) as empty_client:
            totals = {
                endpoint: empty_client.get(f"/research/v1/{endpoint}").json()["meta"]["total"]
                for endpoint in endpoints
            }

        assert totals == {endpoint: 0 for endpoint in endpoints}

    def test_synthetic_fixture_marks_every_formal_looking_entity(self, client):
        entities = {
            "project": client.get("/research/v1/projects/proj-001").json(),
            "data_source": client.get("/research/v1/data-sources/src-001").json(),
            "dataset": client.get("/research/v1/datasets/ds-001").json(),
            "experiment": client.get("/research/v1/experiments/exp-001").json(),
            "run": client.get("/research/v1/runs/run-001").json(),
            "model": client.get("/research/v1/models/mdl-001").json(),
            "prediction": client.get("/research/v1/predictions/pred-001").json(),
            "backtest": client.get("/research/v1/backtests/bt-001").json(),
        }

        assert SYNTHETIC_NON_FORMAL in entities["project"]["tags"]
        assert entities["data_source"]["description"] == SYNTHETIC_NON_FORMAL
        assert entities["dataset"]["description"] == SYNTHETIC_NON_FORMAL
        assert entities["experiment"]["config"]["fixture_mode"] == SYNTHETIC_NON_FORMAL
        assert entities["run"]["params"]["fixture_mode"] == SYNTHETIC_NON_FORMAL
        assert entities["model"]["description"] == SYNTHETIC_NON_FORMAL
        assert entities["prediction"]["metadata"]["fixture_mode"] == SYNTHETIC_NON_FORMAL
        assert entities["backtest"]["name"].startswith(f"[{SYNTHETIC_NON_FORMAL}]")


# ── Error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    def test_not_found_returns_404(self, client):
        resp = client.get("/research/v1/projects/nonexistent")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail["code"] == "NOT_FOUND"
        assert "not found" in detail["message"]

    def test_cors_headers_present(self, client):
        resp = client.options(
            "/research/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is not None

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "SeatFlow Research Engine API"
        assert schema["info"]["version"] == "3.0.0-dev"

    def test_docs_endpoint_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
