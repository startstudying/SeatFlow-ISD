"""Tests for all CRUD route endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seatflow.api.app import create_app


@pytest.fixture()
def client():
    app = create_app(reset_store=True)
    with TestClient(app) as c:
        yield c


# ── Projects ────────────────────────────────────────────────────────────


class TestProjectsRoutes:
    def test_list_projects(self, client):
        resp = client.get("/research/v1/projects")
        assert resp.status_code == 200
        body = resp.json()
        assert body["meta"]["total"] == 3
        assert len(body["items"]) == 3

    def test_list_projects_filtered_by_owner(self, client):
        resp = client.get("/research/v1/projects?owner=seatflow-team")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_get_project(self, client):
        resp = client.get("/research/v1/projects/proj-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "proj-001"
        assert resp.json()["name"] == "Research Project 1"

    def test_get_project_not_found(self, client):
        resp = client.get("/research/v1/projects/missing")
        assert resp.status_code == 404

    def test_create_project(self, client):
        resp = client.post(
            "/research/v1/projects",
            json={"name": "New Project", "owner": "alice", "tags": ["a", "b"]},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "New Project"
        assert body["owner"] == "alice"
        assert body["tags"] == ["a", "b"]
        assert body["id"]
        assert body["status"] == "DRAFT"

    def test_create_project_validates_name(self, client):
        resp = client.post("/research/v1/projects", json={"name": ""})
        assert resp.status_code == 422


# ── Data sources ────────────────────────────────────────────────────────


class TestDataSourcesRoutes:
    def test_list_data_sources(self, client):
        resp = client.get("/research/v1/data-sources")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_filter_data_sources_by_kind(self, client):
        resp = client.get("/research/v1/data-sources?kind=FILE")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert all(d["kind"] == "FILE" for d in items)

    def test_filter_data_sources_by_enabled(self, client):
        resp = client.get("/research/v1/data-sources?enabled=true")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_get_data_source_not_found(self, client):
        resp = client.get("/research/v1/data-sources/missing")
        assert resp.status_code == 404


# ── Datasets ────────────────────────────────────────────────────────────


class TestDatasetsRoutes:
    def test_list_datasets(self, client):
        resp = client.get("/research/v1/datasets")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_filter_datasets_by_status(self, client):
        resp = client.get("/research/v1/datasets?status=READY")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_get_dataset(self, client):
        resp = client.get("/research/v1/datasets/ds-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "ds-001"
        assert resp.json()["row_count"] == 10000

    def test_get_dataset_not_found(self, client):
        resp = client.get("/research/v1/datasets/missing")
        assert resp.status_code == 404


# ── Experiments ─────────────────────────────────────────────────────────


class TestExperimentsRoutes:
    def test_list_experiments(self, client):
        resp = client.get("/research/v1/experiments")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_create_experiment(self, client):
        resp = client.post(
            "/research/v1/experiments",
            json={
                "name": "Exp 99",
                "project_id": "proj-001",
                "config": {"lr": 0.001, "epochs": 10},
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Exp 99"
        assert body["config"] == {"lr": 0.001, "epochs": 10}
        assert body["status"] == "DRAFT"

    def test_get_experiment(self, client):
        resp = client.get("/research/v1/experiments/exp-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "exp-001"

    def test_get_experiment_not_found(self, client):
        resp = client.get("/research/v1/experiments/missing")
        assert resp.status_code == 404

    def test_filter_experiments_by_status(self, client):
        resp = client.get("/research/v1/experiments?status=COMPLETED")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 2


# ── Runs ────────────────────────────────────────────────────────────────


class TestRunsRoutes:
    def test_list_runs(self, client):
        resp = client.get("/research/v1/runs")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_create_run_ok(self, client):
        resp = client.post(
            "/research/v1/runs",
            json={"experiment_id": "exp-001", "params": {"lr": 0.01}},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["experiment_id"] == "exp-001"
        assert body["status"] == "PENDING"
        assert body["params"] == {"lr": 0.01}

    def test_create_run_unknown_experiment_returns_404(self, client):
        resp = client.post(
            "/research/v1/runs",
            json={"experiment_id": "nope"},
        )
        assert resp.status_code == 404

    def test_get_run(self, client):
        resp = client.get("/research/v1/runs/run-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "run-001"

    def test_get_run_not_found(self, client):
        resp = client.get("/research/v1/runs/missing")
        assert resp.status_code == 404


# ── Models ──────────────────────────────────────────────────────────────


class TestModelsRoutes:
    def test_list_models(self, client):
        resp = client.get("/research/v1/models")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_filter_models_by_kind(self, client):
        resp = client.get("/research/v1/models?kind=BASE")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["kind"] == "BASE"

    def test_get_model(self, client):
        resp = client.get("/research/v1/models/mdl-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "mdl-001"

    def test_get_model_not_found(self, client):
        resp = client.get("/research/v1/models/missing")
        assert resp.status_code == 404


# ── Predictions ─────────────────────────────────────────────────────────


class TestPredictionsRoutes:
    def test_list_predictions(self, client):
        resp = client.get("/research/v1/predictions")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_filter_predictions_by_model(self, client):
        resp = client.get("/research/v1/predictions?model_id=mdl-001")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_get_prediction(self, client):
        resp = client.get("/research/v1/predictions/pred-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "pred-001"

    def test_get_prediction_not_found(self, client):
        resp = client.get("/research/v1/predictions/missing")
        assert resp.status_code == 404


# ── Backtests ───────────────────────────────────────────────────────────


class TestBacktestsRoutes:
    def test_list_backtests(self, client):
        resp = client.get("/research/v1/backtests")
        assert resp.status_code == 200
        assert resp.json()["meta"]["total"] == 3

    def test_create_backtest_ok(self, client):
        resp = client.post(
            "/research/v1/backtests",
            json={
                "name": "BT 99",
                "model_id": "mdl-001",
                "dataset_id": "ds-001",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "BT 99"
        assert body["status"] == "PENDING"

    def test_create_backtest_unknown_model(self, client):
        resp = client.post(
            "/research/v1/backtests",
            json={"name": "x", "model_id": "nope", "dataset_id": "ds-001"},
        )
        assert resp.status_code == 404

    def test_create_backtest_unknown_dataset(self, client):
        resp = client.post(
            "/research/v1/backtests",
            json={"name": "x", "model_id": "mdl-001", "dataset_id": "nope"},
        )
        assert resp.status_code == 404

    def test_get_backtest(self, client):
        resp = client.get("/research/v1/backtests/bt-001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "bt-001"

    def test_get_backtest_not_found(self, client):
        resp = client.get("/research/v1/backtests/missing")
        assert resp.status_code == 404
