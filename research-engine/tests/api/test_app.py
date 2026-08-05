"""Tests for the FastAPI application: health check, pagination, error handling."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seatflow.api.app import create_app
from seatflow.api.routes.state import store


@pytest.fixture()
def client():
    """Return a TestClient with a freshly-seeded store."""
    app = create_app(reset_store=True)
    with TestClient(app) as c:
        yield c


# ── Health check ────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_health_returns_up(self, client):
        resp = client.get("/research/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "UP"
        assert "components" in body
        assert "timestamp" in body

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
