"""SeatFlow research engine async worker package.

Exposes the in-memory task store, async task runner, and SSE router so
the API layer can register task-related endpoints and drive long-running
research jobs.
"""

from seatflow.worker.sse import make_tasks_router
from seatflow.worker.task_runner import TaskRunner
from seatflow.worker.task_store import (
    InMemoryTaskStore,
    InvalidStateTransitionError,
    TaskNotFoundError,
    store,
)

__all__ = [
    "InMemoryTaskStore",
    "InvalidStateTransitionError",
    "TaskNotFoundError",
    "TaskRunner",
    "make_tasks_router",
    "store",
]
