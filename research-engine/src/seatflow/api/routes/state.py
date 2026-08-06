"""Shared in-memory store for route handlers.

During development and testing the API is backed by an in-memory store
rather than a real database.  Each entity type has its own ``dict``
keyed by ``id``.  The store is intentionally **not** thread-safe — it
is designed for single-process test usage.

When a persistent storage layer is added, route handlers can be swapped
to use repository interfaces without changing endpoint signatures.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict

from seatflow.api.models import (
    Backtest,
    BacktestStatus,
    DataSource,
    DataSourceKind,
    Dataset,
    DatasetStatus,
    Experiment,
    ExperimentStatus,
    ModelInfo,
    ModelKind,
    Prediction,
    Project,
    ProjectStatus,
    Run,
    RunStatus,
)
from seatflow.domain.time import utc_now

SYNTHETIC_NON_FORMAL = "SYNTHETIC_NON_FORMAL"


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Pre-seeded sample data ──────────────────────────────────────────────


def _seed_projects() -> OrderedDict[str, Project]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        pid = f"proj-{i:03d}"
        items[pid] = Project(
            id=pid,
            name=f"Research Project {i}",
            description=f"Description for project {i}",
            owner="seatflow-team",
            status=ProjectStatus.ACTIVE,
            tags=["alpha", SYNTHETIC_NON_FORMAL],
            created_at=now,
            updated_at=now,
        )
    return items


def _seed_data_sources() -> OrderedDict[str, DataSource]:
    now = utc_now()
    items = OrderedDict()
    for i, kind in enumerate([DataSourceKind.PROVIDER, DataSourceKind.FILE, DataSourceKind.DATABASE], 1):
        sid = f"src-{i:03d}"
        items[sid] = DataSource(
            id=sid,
            name=f"Data Source {i}",
            kind=kind,
            provider="tushare" if kind == DataSourceKind.PROVIDER else None,
            version="v1",
            description=SYNTHETIC_NON_FORMAL,
            enabled=True,
            created_at=now,
        )
    return items


def _seed_datasets() -> OrderedDict[str, Dataset]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        did = f"ds-{i:03d}"
        items[did] = Dataset(
            id=did,
            name=f"Dataset {i}",
            project_id="proj-001",
            source_ids=["src-001", "src-002"],
            status=DatasetStatus.READY,
            row_count=10000 * i,
            description=SYNTHETIC_NON_FORMAL,
            created_at=now,
            updated_at=now,
        )
    return items


def _seed_experiments() -> OrderedDict[str, Experiment]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        eid = f"exp-{i:03d}"
        items[eid] = Experiment(
            id=eid,
            name=f"Experiment {i}",
            project_id="proj-001",
            status=ExperimentStatus.COMPLETED if i < 3 else ExperimentStatus.RUNNING,
            config={"lr": 0.001 * i, "fixture_mode": SYNTHETIC_NON_FORMAL},
            created_at=now,
            updated_at=now,
        )
    return items


def _seed_runs() -> OrderedDict[str, Run]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        rid = f"run-{i:03d}"
        items[rid] = Run(
            id=rid,
            experiment_id="exp-001",
            status=RunStatus.COMPLETED if i < 3 else RunStatus.RUNNING,
            progress=1.0 if i < 3 else 0.5,
            started_at=now,
            completed_at=now if i < 3 else None,
            metrics={"accuracy": 0.85 + 0.01 * i},
            params={"lr": 0.001, "fixture_mode": SYNTHETIC_NON_FORMAL},
            created_at=now,
        )
    return items


def _seed_models() -> OrderedDict[str, ModelInfo]:
    now = utc_now()
    items = OrderedDict()
    for i, kind in enumerate([ModelKind.BASE, ModelKind.DIRECT, ModelKind.SPILLOVER], 1):
        mid = f"mdl-{i:03d}"
        items[mid] = ModelInfo(
            id=mid,
            name=f"Model {i}",
            kind=kind,
            experiment_id="exp-001",
            run_id="run-001",
            version=f"{i}.0",
            description=SYNTHETIC_NON_FORMAL,
            metrics={"f1": 0.8 + 0.02 * i},
            created_at=now,
        )
    return items


def _seed_predictions() -> OrderedDict[str, Prediction]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        pid = f"pred-{i:03d}"
        items[pid] = Prediction(
            id=pid,
            model_id="mdl-001",
            target=f"stock-{i:04d}",
            score=0.7 + 0.05 * i,
            rank=i,
            as_of=now,
            metadata={"fixture_mode": SYNTHETIC_NON_FORMAL},
            created_at=now,
        )
    return items


def _seed_backtests() -> OrderedDict[str, Backtest]:
    now = utc_now()
    items = OrderedDict()
    for i in range(1, 4):
        bid = f"bt-{i:03d}"
        items[bid] = Backtest(
            id=bid,
            name=f"[{SYNTHETIC_NON_FORMAL}] Backtest {i}",
            model_id="mdl-001",
            dataset_id="ds-001",
            status=BacktestStatus.COMPLETED if i < 3 else BacktestStatus.PENDING,
            metrics={"sharpe": 1.2 + 0.1 * i},
            created_at=now,
            completed_at=now if i < 3 else None,
        )
    return items


class _Store:
    """Singleton in-memory store that is empty unless a test fixture is explicit."""

    def __init__(self, *, fixture_mode: str | None = None) -> None:
        self.reset(fixture_mode=fixture_mode)

    def reset(self, *, fixture_mode: str | None = None) -> None:
        """Clear all state, optionally loading the explicit synthetic fixture."""
        if fixture_mode not in (None, SYNTHETIC_NON_FORMAL):
            raise ValueError(f"Unsupported fixture mode: {fixture_mode}")

        self.fixture_mode = fixture_mode
        self.projects: OrderedDict[str, Project] = OrderedDict()
        self.data_sources: OrderedDict[str, DataSource] = OrderedDict()
        self.datasets: OrderedDict[str, Dataset] = OrderedDict()
        self.experiments: OrderedDict[str, Experiment] = OrderedDict()
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.models: OrderedDict[str, ModelInfo] = OrderedDict()
        self.predictions: OrderedDict[str, Prediction] = OrderedDict()
        self.backtests: OrderedDict[str, Backtest] = OrderedDict()

        if fixture_mode == SYNTHETIC_NON_FORMAL:
            self.projects = _seed_projects()
            self.data_sources = _seed_data_sources()
            self.datasets = _seed_datasets()
            self.experiments = _seed_experiments()
            self.runs = _seed_runs()
            self.models = _seed_models()
            self.predictions = _seed_predictions()
            self.backtests = _seed_backtests()


# Module-level singleton — imported by route modules.
store = _Store()
