"""In-memory task store for the SeatFlow research engine.

Provides:

* CRUD operations on tasks keyed by ``task_id``
* A lifecycle state machine enforcing the transition rules in the OpenAPI
  contract (see :class:`seatflow.api.models.TaskStatus`)
* An append-only event log per task, used by the SSE endpoint to stream
  progress updates

The store is intentionally **not** process-safe.  Production deployments
swap in a Redis-backed implementation behind the same interface.
"""

from __future__ import annotations

import asyncio
import builtins
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Coroutine, Deque, Optional

from seatflow.api.models import ErrorResponse, TaskEvent, TaskState, TaskStatus
from seatflow.domain.time import utc_now


# ── Allowed state transitions ──────────────────────────────────────────
# Mirrors the contract in ``contracts/openapi/research-api.yaml``.

_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.REQUESTED: frozenset(
        {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.BLOCKED: frozenset(
        {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.CANCELLED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}

_TERMINAL_STATES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED}
)


class TaskNotFoundError(LookupError):
    """Raised when a task id is not present in the store."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task '{task_id}' not found")
        self.task_id = task_id


class InvalidStateTransitionError(ValueError):
    """Raised when a requested state transition violates the lifecycle."""

    def __init__(self, task_id: str, current: TaskStatus, target: TaskStatus) -> None:
        super().__init__(
            f"Invalid transition for task '{task_id}': {current.value} -> {target.value}"
        )
        self.task_id = task_id
        self.current = current
        self.target = target


# ── Store ──────────────────────────────────────────────────────────────


class InMemoryTaskStore:
    """In-memory task store with event log and pub/sub hooks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._events: dict[str, Deque[TaskEvent]] = defaultdict(deque)
        # Listeners awaiting new events (one per task_id).  Each listener is
        # an :class:`asyncio.Queue` populated by :meth:`append_event`.
        self._listeners: dict[str, set[asyncio.Queue[TaskEvent]]] = defaultdict(set)
        # Coarse lock used only to protect list-mutation operations on
        # the listener sets; the underlying ``_tasks`` dict is not
        # guaranteed to be concurrent-safe.
        self._lock = asyncio.Lock()
        # Background coroutines spawned for each task (test introspection).
        self._background: dict[str, asyncio.Task[None]] = {}

    # ── Lifecycle helpers ──────────────────────────────────────────────

    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex

    def _now(self) -> Any:
        return utc_now()

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(
        self,
        *,
        kind: str,
        spec: dict[str, Any],
        status: TaskStatus = TaskStatus.REQUESTED,
    ) -> TaskState:
        """Create a new task in the ``REQUESTED`` state (or *status*)."""
        task_id = self._gen_id()
        now = self._now()
        state = TaskState(
            taskId=task_id,
            status=status,
            progress=0.0,
            updatedAt=now,
        )
        self._tasks[task_id] = state
        self._append(
            task_id,
            event_type="TASK_CREATED",
            payload={"status": status.value, "kind": kind, "spec": spec},
        )
        return state

    def get(self, task_id: str) -> TaskState:
        """Return the current state for *task_id* or raise :class:`TaskNotFoundError`."""
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(task_id) from exc

    def list(
        self,
        *,
        status: Optional[TaskStatus] = None,
        kind: Optional[str] = None,
    ) -> list[TaskState]:
        """Return all tasks, optionally filtered by status and/or kind."""
        # ``kind`` is metadata stored in the create event payload; we
        # recover it lazily from the first event of each task.
        results: list[TaskState] = []
        for task_id, state in self._tasks.items():
            if status is not None and state.status != status:
                continue
            if kind is not None:
                if not self._events[task_id]:
                    continue
                payload = self._events[task_id][0].payload
                if payload.get("kind") != kind:
                    continue
            results.append(state)
        return results

    def delete(self, task_id: str) -> None:
        """Remove a task and all of its events (test helper)."""
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        del self._tasks[task_id]
        self._events.pop(task_id, None)
        self._listeners.pop(task_id, None)

    # ── State transitions ─────────────────────────────────────────────

    def transition(
        self,
        task_id: str,
        target: TaskStatus,
        *,
        progress: Optional[float] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[ErrorResponse] = None,
        message: Optional[str] = None,
    ) -> TaskState:
        """Transition *task_id* to *target*, enforcing the lifecycle.

        Emits the appropriate ``TASK_*`` event.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        current = self._tasks[task_id].status
        if target not in _ALLOWED_TRANSITIONS[current]:
            raise InvalidStateTransitionError(task_id, current, target)
        now = self._now()
        updated = TaskState(
            taskId=task_id,
            status=target,
            progress=progress if progress is not None else self._tasks[task_id].progress,
            updatedAt=now,
            result=result,
            error=error,
        )
        self._tasks[task_id] = updated

        # Pick the canonical event type for the target state.
        event_type = {
            TaskStatus.QUEUED: "TASK_QUEUED",
            TaskStatus.RUNNING: "TASK_STARTED",
            TaskStatus.COMPLETED: "TASK_COMPLETED",
            TaskStatus.CANCELLED: "TASK_CANCELLED",
            TaskStatus.FAILED: "TASK_FAILED",
            TaskStatus.BLOCKED: "TASK_BLOCKED",
            TaskStatus.REQUESTED: "TASK_CREATED",
        }[target]
        payload: dict[str, Any] = {"status": target.value}
        if message is not None:
            payload["message"] = message
        if progress is not None:
            payload["progress"] = progress
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = {
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "retryable": error.retryable,
                "remediation": error.remediation,
            }
        self._append(task_id, event_type=event_type, payload=payload)
        return updated

    def progress(
        self,
        task_id: str,
        progress: float,
        *,
        message: Optional[str] = None,
    ) -> TaskState:
        """Emit a progress event for a running task.

        Does **not** change the task status; the task is expected to be
        in ``RUNNING`` already.
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        state = self._tasks[task_id]
        if state.status in _TERMINAL_STATES:
            raise InvalidStateTransitionError(task_id, state.status, state.status)
        updated = TaskState(
            taskId=task_id,
            status=state.status,
            progress=progress,
            updatedAt=self._now(),
            result=state.result,
            error=state.error,
        )
        self._tasks[task_id] = updated
        payload: dict[str, Any] = {"status": state.status.value, "progress": progress}
        if message is not None:
            payload["message"] = message
        self._append(task_id, event_type="TASK_PROGRESS", payload=payload)
        return updated

    def cancel(self, task_id: str) -> TaskState:
        """Cancel a task if it is not yet in a terminal state."""
        state = self.get(task_id)
        if state.status in _TERMINAL_STATES:
            raise InvalidStateTransitionError(task_id, state.status, TaskStatus.CANCELLED)
        return self.transition(task_id, TaskStatus.CANCELLED, message="Cancelled by user")

    def is_terminal(self, status: TaskStatus) -> bool:
        """Return ``True`` if *status* is a terminal state."""
        return status in _TERMINAL_STATES

    # ── Event log ─────────────────────────────────────────────────────

    def events(self, task_id: str) -> builtins.list[TaskEvent]:
        """Return a copy of the event log for *task_id*."""
        return list(self._events.get(task_id, ()))

    def _append(
        self,
        task_id: str,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        event = TaskEvent(
            eventType=event_type,
            taskId=task_id,
            timestamp=self._now(),
            payload=payload,
        )
        self._events[task_id].append(event)
        # Fan out to any active listeners.
        for queue in list(self._listeners.get(task_id, ())):
            queue.put_nowait(event)

    # ── Pub/sub for SSE ───────────────────────────────────────────────

    async def subscribe(self, task_id: str) -> asyncio.Queue[TaskEvent]:
        """Register a new listener queue for *task_id*."""
        queue: asyncio.Queue[TaskEvent] = asyncio.Queue()
        async with self._lock:
            self._listeners[task_id].add(queue)
        return queue

    async def unsubscribe(self, task_id: str, queue: asyncio.Queue[TaskEvent]) -> None:
        """Remove a previously-registered listener queue."""
        async with self._lock:
            self._listeners[task_id].discard(queue)

    # ── Introspection helpers (mostly for tests) ───────────────────────

    def register_background(
        self,
        task_id: str,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Track a background coroutine for *task_id* (test-only)."""
        self._background[task_id] = asyncio.create_task(coro_factory())

    async def wait_background(self, task_id: str, timeout: float = 5.0) -> None:
        """Await a tracked background coroutine, ignoring timeouts."""
        bg = self._background.get(task_id)
        if bg is None:
            return
        try:
            await asyncio.wait_for(bg, timeout=timeout)
        except asyncio.TimeoutError:
            bg.cancel()


# Module-level singleton store.  Tests should construct their own
# :class:`InMemoryTaskStore` instances; this exists for the default
# router exported by ``seatflow.worker.sse``.
store = InMemoryTaskStore()
