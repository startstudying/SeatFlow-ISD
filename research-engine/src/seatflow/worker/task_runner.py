"""Async task runner for the SeatFlow research engine.

The runner consumes queued tasks and drives them through the
:class:`seatflow.api.models.TaskStatus` lifecycle by calling the
appropriate handler.  Handlers are registered by *kind* and return a
result payload that is recorded on the task state.

Queue integration:

By default the runner uses an in-process :class:`asyncio.Queue` (ideal
for tests).  When ``redis_url`` is provided, a thin Redis-backed FIFO
queue is used — this satisfies the "Redis-based task queue" requirement
without forcing a live Redis dependency during unit tests.

Supported task kinds:

* ``experiment_run`` — delegate an experiment run to a configured pipeline
* ``backtest`` — delegate a backtest replay to a configured pipeline
* ``prediction`` — delegate a daily inference job to a configured pipeline
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import uuid
from typing import Any, Awaitable, Callable, Optional, Protocol

from seatflow.api.models import ErrorResponse, TaskState, TaskStatus
from seatflow.worker.task_store import (
    InMemoryTaskStore,
    InvalidStateTransitionError,
    TaskNotFoundError,
)

logger = logging.getLogger(__name__)


# ── Redis-backed queue (lazy, optional) ────────────────────────────────


class RedisQueue:
    """Minimal async FIFO queue backed by Redis lists.

    Uses blocking-pop semantics so :meth:`get` awaits until an item is
    available.  This is a thin wrapper; production deployments may swap
    in a richer implementation (priorities, dead-letter, etc.).
    """

    def __init__(self, redis_url: str, queue_name: str = "seatflow:tasks") -> None:
        import redis.asyncio as aioredis  # local import to keep tests lean

        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._queue_name = queue_name

    async def put(self, task_id: str) -> None:
        await self._client.rpush(self._queue_name, task_id)

    async def get(self) -> str:
        # ``blpop`` returns ``(key, value)`` or ``None`` on timeout.
        result = await self._client.blpop(self._queue_name, timeout=1)
        if result is None:
            raise asyncio.TimeoutError
        _key, value = result
        return value.decode() if isinstance(value, bytes) else value

    async def close(self) -> None:
        await self._client.aclose()


# ── Handler registry ───────────────────────────────────────────────────


HandlerResult = dict[str, Any]
TaskHandler = Callable[[dict[str, Any], "TaskRunner"], Awaitable[HandlerResult]]
ProgressReporter = Callable[[float, Optional[str]], None]


class TaskBlockedError(RuntimeError):
    """A recoverable research-capability blocker with machine-readable guidance."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        remediation: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.details = details


class ResearchPipelineAdapter(Protocol):
    """Boundary implemented by a real experiment/backtest/prediction pipeline."""

    async def execute(
        self,
        *,
        task_kind: str,
        spec: dict[str, Any],
        report_progress: ProgressReporter,
    ) -> HandlerResult:
        """Execute one research task and return a lineage-bearing result."""

    def readiness(self) -> dict[str, Any]:
        """Return an explicit synchronous readiness report."""


def pipeline_capability(
    pipeline_adapter: Optional[ResearchPipelineAdapter],
) -> dict[str, str]:
    """Describe readiness without conflating process health and research capability."""
    if pipeline_adapter is None:
        return {
            "status": "BLOCKED",
            "code": "PIPELINE_ADAPTER_UNAVAILABLE",
            "remediation": (
                "Configure a real ResearchPipelineAdapter before running research tasks."
            ),
        }
    execute = getattr(pipeline_adapter, "execute", None)
    if not callable(execute) or not inspect.iscoroutinefunction(execute):
        return {
            "status": "BLOCKED",
            "code": "PIPELINE_ADAPTER_INVALID",
            "remediation": "Configure an adapter with an async execute() method.",
        }

    readiness = getattr(pipeline_adapter, "readiness", None)
    if not callable(readiness) or inspect.iscoroutinefunction(readiness):
        return {
            "status": "BLOCKED",
            "code": "PIPELINE_READINESS_UNVERIFIED",
            "remediation": (
                "Implement a synchronous readiness() check on the pipeline adapter."
            ),
        }

    try:
        report = readiness()
    except Exception:  # noqa: BLE001
        logger.exception("Pipeline adapter readiness check failed")
        return {
            "status": "BLOCKED",
            "code": "PIPELINE_READINESS_CHECK_FAILED",
            "remediation": "Repair the adapter readiness check before serving research tasks.",
        }

    if not isinstance(report, dict) or not isinstance(report.get("ready"), bool):
        return {
            "status": "BLOCKED",
            "code": "PIPELINE_READINESS_INVALID",
            "remediation": "Return a readiness report containing a boolean ready field.",
        }
    if report["ready"] is not True:
        code = report.get("code")
        remediation = report.get("remediation")
        return {
            "status": "BLOCKED",
            "code": code if isinstance(code, str) and code else "PIPELINE_NOT_READY",
            "remediation": (
                remediation
                if isinstance(remediation, str) and remediation
                else "Resolve the adapter readiness blocker before serving research tasks."
            ),
        }
    return {"status": "UP", "adapter": type(pipeline_adapter).__name__}


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_LINEAGE_FIELDS: dict[str, frozenset[str]] = {
    "experiment_run": frozenset(
        {"dataset_manifest_sha256", "model_artifact_sha256"}
    ),
    "backtest": frozenset(
        {
            "dataset_manifest_sha256",
            "model_artifact_sha256",
            "backtest_artifact_sha256",
        }
    ),
    "prediction": frozenset(
        {
            "feature_manifest_sha256",
            "model_artifact_sha256",
            "prediction_artifact_sha256",
        }
    ),
}


def _invalid_lineage_fields(task_kind: str, lineage: dict[str, Any]) -> dict[str, Any]:
    required = _REQUIRED_LINEAGE_FIELDS[task_kind]
    missing = sorted(required.difference(lineage))
    malformed = sorted(
        field
        for field in required.intersection(lineage)
        if not isinstance(lineage[field], str)
        or _SHA256_PATTERN.fullmatch(lineage[field]) is None
    )
    return {
        "task_kind": task_kind,
        "required_fields": sorted(required),
        "missing_fields": missing,
        "invalid_fields": malformed,
    }


async def _pipeline_handler(
    task_kind: str,
    spec: dict[str, Any],
    runner: "TaskRunner",
) -> HandlerResult:
    adapter = runner.pipeline_adapter
    capability = pipeline_capability(adapter)
    if capability["status"] != "UP":
        code = capability["code"]
        raise TaskBlockedError(
            code=code,
            message=(
                "Research pipeline adapter is not configured"
                if code == "PIPELINE_ADAPTER_UNAVAILABLE"
                else "Research pipeline adapter is not ready"
            ),
            remediation=capability["remediation"],
            details={"task_kind": task_kind},
        )
    assert adapter is not None

    def report_progress(progress: float, message: Optional[str] = None) -> None:
        runner.store.progress(runner.current_task_id, progress, message=message)

    result = await adapter.execute(
        task_kind=task_kind,
        spec=spec,
        report_progress=report_progress,
    )
    lineage = result.get("artifact_lineage")
    if not isinstance(lineage, dict) or not lineage:
        raise TaskBlockedError(
            code="ARTIFACT_LINEAGE_MISSING",
            message="Pipeline result has no immutable artifact lineage",
            remediation=(
                "Persist immutable input and output artifact identifiers before completion."
            ),
            details={"task_kind": task_kind},
        )
    invalid = _invalid_lineage_fields(task_kind, lineage)
    if invalid["missing_fields"] or invalid["invalid_fields"]:
        raise TaskBlockedError(
            code="ARTIFACT_LINEAGE_INVALID",
            message="Pipeline result has invalid task-specific artifact lineage",
            remediation=(
                "Return every required immutable artifact SHA-256 for this task kind."
            ),
            details=invalid,
        )
    return result


async def _experiment_run_handler(
    spec: dict[str, Any], runner: "TaskRunner"
) -> HandlerResult:
    return await _pipeline_handler("experiment_run", spec, runner)


async def _backtest_handler(spec: dict[str, Any], runner: "TaskRunner") -> HandlerResult:
    return await _pipeline_handler("backtest", spec, runner)


async def _prediction_handler(spec: dict[str, Any], runner: "TaskRunner") -> HandlerResult:
    return await _pipeline_handler("prediction", spec, runner)


_DEFAULT_HANDLERS: dict[str, TaskHandler] = {
    "experiment_run": _experiment_run_handler,
    "backtest": _backtest_handler,
    "prediction": _prediction_handler,
}


# ── Runner ─────────────────────────────────────────────────────────────


class TaskRunner:
    """Async task runner that drives queued tasks to completion."""

    def __init__(
        self,
        store: InMemoryTaskStore,
        *,
        handlers: Optional[dict[str, TaskHandler]] = None,
        pipeline_adapter: Optional[ResearchPipelineAdapter] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self.store = store
        self.handlers: dict[str, TaskHandler] = {**_DEFAULT_HANDLERS, **(handlers or {})}
        self.pipeline_adapter = pipeline_adapter
        self._redis_url = redis_url
        self._redis: Optional[RedisQueue] = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task[None]] = None
        # The id of the task currently being processed (set during a handler
        # call so progress helpers can locate it).
        self.current_task_id: str = ""

    # ── Queue helpers ─────────────────────────────────────────────────

    async def _ensure_queue(self) -> None:
        if self._redis_url is not None and self._redis is None:
            self._redis = RedisQueue(self._redis_url)

    async def _enqueue(self, task_id: str) -> None:
        await self._ensure_queue()
        if self._redis is not None:
            await self._redis.put(task_id)
        else:
            await self._queue.put(task_id)

    async def _dequeue(self) -> str:
        if self._redis is not None:
            while self._running:
                try:
                    return await self._redis.get()
                except asyncio.TimeoutError:
                    continue
            raise asyncio.CancelledError
        return await self._queue.get()

    # ── Public API ────────────────────────────────────────────────────

    def register_handler(self, kind: str, handler: TaskHandler) -> None:
        """Register or replace a handler for *kind*."""
        self.handlers[kind] = handler

    async def submit(
        self,
        *,
        kind: str,
        spec: Optional[dict[str, Any]] = None,
    ) -> TaskState:
        """Create a task, queue it, and return its initial state.

        Raises :class:`ValueError` if *kind* has no registered handler.
        """
        if kind not in self.handlers:
            raise ValueError(f"No handler registered for task kind '{kind}'")
        spec = spec or {}
        state = self.store.create(kind=kind, spec=spec)
        # Move from REQUESTED -> QUEUED.
        self.store.transition(state.task_id, TaskStatus.QUEUED)
        await self._enqueue(state.task_id)
        return self.store.get(state.task_id)

    async def start(self) -> None:
        """Start the background worker loop (idempotent)."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Signal the worker loop to stop and await its exit."""
        self._running = False
        if self._redis is None:
            # Nudge the in-process queue so a blocked ``get`` returns.
            await self._queue.put("")
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
            self._worker_task = None
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def run_once(self) -> Optional[TaskState]:
        """Process at most one queued task and return its final state.

        Useful for deterministic tests that do not want to spawn the
        background worker loop.
        """
        await self._ensure_queue()
        task_id = await self._dequeue()
        if not task_id:
            return None
        return await self._process(task_id)

    # ── Worker loop ───────────────────────────────────────────────────

    async def _run(self) -> None:
        while self._running:
            try:
                task_id = await self._dequeue()
            except asyncio.CancelledError:
                break
            if not task_id:
                continue
            try:
                await self._process(task_id)
            except Exception:  # noqa: BLE001
                logger.exception("Worker loop error processing task %s", task_id)

    async def _process(self, task_id: str) -> Optional[TaskState]:
        """Drive a single task from QUEUED through to a terminal state."""
        try:
            state = self.store.get(task_id)
        except TaskNotFoundError:
            logger.warning("Task %s vanished before processing", task_id)
            return None

        if state.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            # Already terminal (e.g. cancelled while queued).  Nothing to do.
            return state

        # Recover the kind/spec from the creation event.
        events = self.store.events(task_id)
        create_event = events[0] if events else None
        payload = create_event.payload if create_event else {}
        kind = payload.get("kind", "")
        spec = payload.get("spec", {})

        handler = self.handlers.get(kind)
        if handler is None:
            # Per lifecycle contract only RUNNING -> FAILED is allowed, so
            # advance to RUNNING before recording the failure.
            self.store.transition(task_id, TaskStatus.RUNNING, message="Task started")
            err = ErrorResponse.build(
                code="NO_HANDLER",
                message=f"No handler registered for task kind '{kind}'",
            )
            return self.store.transition(
                task_id, TaskStatus.FAILED, error=err, message=err.message
            )

        self.store.transition(task_id, TaskStatus.RUNNING, message="Task started")
        self.current_task_id = task_id
        try:
            result = await handler(spec, self)
        except TaskBlockedError as exc:
            err = ErrorResponse.build(
                code=exc.code,
                message=str(exc),
                details=exc.details,
                retryable=False,
                remediation=exc.remediation,
            )
            return self.store.transition(
                task_id,
                TaskStatus.BLOCKED,
                error=err,
                message=err.message,
            )
        except Exception:  # noqa: BLE001
            trace_id = uuid.uuid4().hex
            logger.exception(
                "Task handler failed trace_id=%s task_id=%s kind=%s",
                trace_id,
                task_id,
                kind,
            )
            err = ErrorResponse.build(
                code="TASK_FAILED",
                message="Research task failed",
                details={"task_kind": kind},
                trace_id=trace_id,
                retryable=False,
                remediation="Contact the service operator with the traceId.",
            )
            return self.store.transition(
                task_id, TaskStatus.FAILED, error=err, message=err.message
            )
        finally:
            self.current_task_id = ""

        try:
            return self.store.transition(
                task_id,
                TaskStatus.COMPLETED,
                progress=1.0,
                result=result,
                message="Task completed",
            )
        except InvalidStateTransitionError:
            # The task may have been cancelled mid-flight; return current state.
            return self.store.get(task_id)
