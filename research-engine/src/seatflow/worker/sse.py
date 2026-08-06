"""Server-Sent Events endpoint for task progress.

Exposes ``GET /research/v1/tasks/{task_id}/events`` which streams
``TaskEvent`` payloads as SSE messages until the task reaches a
terminal state (or the client disconnects).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from seatflow.api.models import TaskEvent, TaskState
from seatflow.worker.task_store import InMemoryTaskStore, TaskNotFoundError

router = APIRouter(prefix="/research/v1/tasks", tags=["tasks"])


def _format_sse(event: TaskEvent) -> str:
    """Serialise a :class:`TaskEvent` into an SSE ``data:`` frame."""
    payload = event.model_dump(mode="json", by_alias=True)
    return f"event: {event.event_type}\ndata: {json.dumps(payload)}\n\n"


def make_tasks_router(store: InMemoryTaskStore) -> APIRouter:
    """Build a tasks router bound to *store*.

    Keeping this a factory lets tests construct isolated stores and
    routers without mutating module-level singletons.
    """
    local_router = APIRouter(prefix="/research/v1/tasks", tags=["tasks"])

    @local_router.get("/{task_id}", name="get_task")
    def get_task(task_id: str) -> TaskState:
        try:
            return store.get(task_id)
        except TaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"Task '{task_id}' not found"},
            )

    @local_router.get("/{task_id}/events", name="stream_task_events")
    async def stream_task_events(
        task_id: str, request: Request
    ) -> StreamingResponse:
        """Stream task events as Server-Sent Events.

        Replays the existing event log and then streams live events
        until the task reaches a terminal state or the client
        disconnects.
        """
        try:
            store.get(task_id)
        except TaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Task '{task_id}' not found",
                },
            )

        async def event_stream() -> AsyncIterator[str]:
            # Replay buffered events first.
            replayed: list[TaskEvent] = list(store.events(task_id))
            terminal_seen = False
            for evt in replayed:
                yield _format_sse(evt)
                if _is_terminal_event(evt.event_type):
                    terminal_seen = True
            if terminal_seen:
                return

            # Subscribe for live events.
            queue = await store.subscribe(task_id)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        # Send a comment keep-alive frame.
                        yield ": keep-alive\n\n"
                        continue
                    yield _format_sse(evt)
                    if _is_terminal_event(evt.event_type):
                        break
            finally:
                await store.unsubscribe(task_id, queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @local_router.post("/{task_id}/cancel", name="cancel_task")
    async def cancel_task(task_id: str) -> TaskState:
        try:
            store.get(task_id)
        except TaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"Task '{task_id}' not found",
                },
            )
        try:
            return store.cancel(task_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "INVALID_TRANSITION",
                    "message": str(exc),
                },
            )

    return local_router


def _is_terminal_event(event_type: str) -> bool:
    return event_type in {
        "TASK_COMPLETED",
        "TASK_CANCELLED",
        "TASK_FAILED",
    }


# Default module-level router bound to a singleton store.  Tests should
# prefer :func:`make_tasks_router` with an isolated store.
from seatflow.worker.task_store import store as default_store  # noqa: E402

default_router = make_tasks_router(default_store)
