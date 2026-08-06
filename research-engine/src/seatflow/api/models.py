"""Pydantic v2 models for SeatFlow research API.

This module defines DTOs for request/response shapes for every endpoint.
Conventions:

* All timestamps are timezone-aware UTC ``datetime`` objects; FastAPI serialises
  them as ISO 8601 strings.
* Frozen Pydantic models (``model_config = {"frozen": True}``) are used for
  response DTOs to prevent accidental mutation after construction.
* Mutating inputs (POST bodies) use mutable models so request handlers can
  derive domain objects from them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from seatflow.domain.time import utc_now

T = TypeVar("T")


# ── Enumerations ────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Lifecycle states for long-running research tasks.

    Allowed transitions (per OpenAPI contract):

    * ``REQUESTED -> QUEUED -> RUNNING -> COMPLETED``
    * ``REQUESTED | QUEUED | RUNNING -> CANCELLED``
    * ``RUNNING -> FAILED``
    * ``REQUESTED | QUEUED | RUNNING -> BLOCKED``

    Terminal states (``COMPLETED``, ``CANCELLED``, ``FAILED``, ``BLOCKED``)
    do not transition.
    """

    REQUESTED = "REQUESTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class HealthState(str, Enum):
    """Aggregate health probe state."""

    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"


class ProjectStatus(str, Enum):
    """Top-level research project status."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DRAFT = "DRAFT"


class DataSourceKind(str, Enum):
    """Categories of upstream data sources."""

    PROVIDER = "PROVIDER"
    FILE = "FILE"
    DATABASE = "DATABASE"
    STREAM = "STREAM"


class DatasetStatus(str, Enum):
    """Lifecycle status of a dataset artifact."""

    DRAFT = "DRAFT"
    BUILDING = "BUILDING"
    READY = "READY"
    FAILED = "FAILED"


class ExperimentStatus(str, Enum):
    """Status of an experiment definition."""

    DRAFT = "DRAFT"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelKind(str, Enum):
    """Kind of model stored in the registry."""

    BASE = "BASE"
    DIRECT = "DIRECT"
    SPILLOVER = "SPILLOVER"
    FULL = "FULL"


class BacktestStatus(str, Enum):
    """Status of a backtest replay run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ── Error envelope ──────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error envelope returned for non-2xx responses.

    Matches the OpenAPI ``SeatFlowError`` schema (sans ``traceId`` which is
    injected per request by the application layer).
    """

    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Optional structured error context"
    )
    trace_id: Optional[str] = Field(
        default=None, alias="traceId", description="Distributed trace identifier"
    )
    timestamp: datetime = Field(
        default_factory=utc_now, description="Time at which the error was generated"
    )
    retryable: bool = Field(default=False, description="Whether the caller can retry")
    remediation: Optional[str] = Field(
        default=None, description="Hint for resolving the error"
    )

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @classmethod
    def build(
        cls,
        code: str,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        retryable: bool = False,
        remediation: Optional[str] = None,
    ) -> "ErrorResponse":
        """Construct an :class:`ErrorResponse` with a fresh UTC timestamp."""
        return cls(
            code=code,
            message=message,
            details=details,
            traceId=trace_id,
            timestamp=utc_now(),
            retryable=retryable,
            remediation=remediation,
        )


# ── Pagination ──────────────────────────────────────────────────────────


class PaginationParams(BaseModel):
    """Standard query parameters for paginated list endpoints."""

    page: int = Field(default=1, ge=1, description="1-indexed page number")
    page_size: int = Field(
        default=50, ge=1, le=500, description="Number of items per page"
    )

    @property
    def offset(self) -> int:
        """Return the SQL-style offset corresponding to ``page``/``page_size``."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Return the page size (alias for ``page_size``)."""
        return self.page_size


class PageMeta(BaseModel):
    """Pagination metadata for a response payload."""

    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool = Field(default=False, alias="hasNext")
    has_prev: bool = Field(default=False, alias="hasPrev")

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated wrapper used by every list endpoint."""

    items: list[T]
    meta: PageMeta

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_items(
        cls,
        items: list[T],
        *,
        page: int,
        page_size: int,
        total: int,
    ) -> "PaginatedResponse[T]":
        """Build a paginated envelope from a slice and the total count."""
        total_pages = (total + page_size - 1) // page_size if page_size else 0
        meta = PageMeta(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            hasNext=page < total_pages,
            hasPrev=page > 1,
        )
        return cls(items=items, meta=meta)


# ── Domain DTOs (frozen responses) ──────────────────────────────────────


class Project(BaseModel):
    """A research project grouping experiments, datasets, and runs."""

    id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    status: ProjectStatus = ProjectStatus.DRAFT
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True, use_enum_values=False)


class DataSource(BaseModel):
    """An upstream data source feeding the research engine."""

    id: str
    name: str
    kind: DataSourceKind
    provider: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class Dataset(BaseModel):
    """A constructed research dataset."""

    id: str
    name: str
    project_id: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    status: DatasetStatus = DatasetStatus.DRAFT
    row_count: Optional[int] = None
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class Experiment(BaseModel):
    """A research experiment definition."""

    id: str
    name: str
    project_id: Optional[str] = None
    description: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class Run(BaseModel):
    """A single execution of an experiment."""

    id: str
    experiment_id: str
    status: RunStatus = RunStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class ModelInfo(BaseModel):
    """Metadata about a trained model artifact."""

    id: str
    name: str
    kind: ModelKind
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    version: str = "1"
    description: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class Prediction(BaseModel):
    """A model prediction for a target entity at a given time."""

    id: str
    model_id: str
    target: str
    score: float
    rank: Optional[int] = None
    horizon: Optional[str] = None
    as_of: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


class Backtest(BaseModel):
    """A backtest replay run."""

    id: str
    name: str
    model_id: str
    dataset_id: str
    status: BacktestStatus = BacktestStatus.PENDING
    range_from: Optional[datetime] = None
    range_to: Optional[datetime] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(frozen=True)


# ── Task-related DTOs (mirror OpenAPI TaskState/TaskHandle) ─────────────


class TaskEvent(BaseModel):
    """SSE event payload for task lifecycle transitions."""

    event_type: str = Field(..., alias="eventType")
    task_id: str = Field(..., alias="taskId")
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = Field(default=None, alias="traceId")

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class TaskHandle(BaseModel):
    """Initial handle returned when a long-running operation is accepted."""

    task_id: str = Field(..., alias="taskId")
    status: TaskStatus
    accepted_at: datetime = Field(..., alias="acceptedAt")
    poll_url: Optional[str] = Field(default=None, alias="pollUrl")
    stream_url: Optional[str] = Field(default=None, alias="streamUrl")

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class TaskState(BaseModel):
    """Current state of a long-running task."""

    task_id: str = Field(..., alias="taskId")
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: datetime = Field(..., alias="updatedAt")
    result: Optional[dict[str, Any]] = None
    error: Optional[ErrorResponse] = None

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class TaskPage(BaseModel):
    """Page of task states used by ``/tasks`` listing endpoints."""

    items: list[TaskState]
    meta: PageMeta

    model_config = ConfigDict(frozen=True)


# ── Health ──────────────────────────────────────────────────────────────


class HealthReport(BaseModel):
    """Aggregate health report returned by the liveness/readiness probe."""

    status: HealthState = HealthState.UP
    components: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)

    model_config = ConfigDict(frozen=True)


# ── Mutable request bodies ─────────────────────────────────────────────


class ProjectCreate(BaseModel):
    """Request body for creating a project."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class ExperimentCreate(BaseModel):
    """Request body for creating an experiment."""

    name: str = Field(..., min_length=1, max_length=200)
    project_id: Optional[str] = None
    description: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)


class RunCreate(BaseModel):
    """Request body for creating a run."""

    experiment_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class BacktestCreate(BaseModel):
    """Request body for creating a backtest."""

    name: str = Field(..., min_length=1, max_length=200)
    model_id: str
    dataset_id: str
    range_from: Optional[datetime] = None
    range_to: Optional[datetime] = None


__all__ = [
    "Backtest",
    "BacktestCreate",
    "BacktestStatus",
    "DataSource",
    "DataSourceKind",
    "Dataset",
    "DatasetStatus",
    "ErrorResponse",
    "Experiment",
    "ExperimentCreate",
    "ExperimentStatus",
    "HealthReport",
    "HealthState",
    "ModelInfo",
    "ModelKind",
    "PageMeta",
    "PaginatedResponse",
    "PaginationParams",
    "Prediction",
    "Project",
    "ProjectCreate",
    "ProjectStatus",
    "Run",
    "RunCreate",
    "RunStatus",
    "TaskEvent",
    "TaskHandle",
    "TaskPage",
    "TaskState",
    "TaskStatus",
]
