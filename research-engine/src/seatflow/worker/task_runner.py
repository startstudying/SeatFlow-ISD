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

* ``experiment_run`` — simulate an experiment run
* ``backtest`` — simulate a backtest replay
* ``prediction`` — simulate a daily inference/prediction job
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

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
        return value

    async def close(self) -> None:
        await self._client.aclose()


# ── Handler registry ───────────────────────────────────────────────────


HandlerResult = dict[str, Any]
TaskHandler = Callable[[dict[str, Any], "TaskRunner"], Awaitable[HandlerResult]]


async def _experiment_run_handler(spec: dict[str, Any], runner: "TaskRunner") -> HandlerResult:
    """Simulate an experiment run with staged progress."""
    experiment_id = spec.get("experiment_id", "exp-000")
    for step in range(1, 5):
        await asyncio.sleep(0)
        runner.store.progress(
            runner.current_task_id,
            step / 4,
            message=f"training step {step}/4",
        )
    return {"experiment_id": experiment_id, "metrics": {"accuracy": 0.9}}


async def _backtest_handler(spec: dict[str, Any], runner: "TaskRunner") -> HandlerResult:
    """Simulate a backtest replay with staged progress."""
    for step in range(1, 4):
        await asyncio.sleep(0)
        runner.store.progress(
            runner.current_task_id,
            step / 3,
            message=f"replaying step {step}/3",
        )
    return {"sharpe": 1.3, "equity_curve": []}


async def _prediction_handler(spec: dict[str, Any], runner: "TaskRunner") -> HandlerResult:
    """Simulate a daily inference/prediction job."""
    for step in range(1, 3):
        await asyncio.sleep(0)
        runner.store.progress(
            runner.current_task_id,
            step / 2,
            message=f"inferring step {step}/2",
        )
    return {"predictions": [{"target": "stock-0001", "score": 0.88}]}


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
        redis_url: Optional[str] = None,
    ) -> None:
        self.store = store
        self.handlers: dict[str, TaskHandler] = {**_DEFAULT_HANDLERS, **(handlers or {})}
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
        except Exception as exc:  # noqa: BLE001
            err = ErrorResponse.build(
                code="TASK_FAILED",
                message=str(exc),
                retryable=False,
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
