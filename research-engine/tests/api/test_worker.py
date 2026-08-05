"""Tests for the async task runner and in-memory task store."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from seatflow.api.models import ErrorResponse, TaskStatus
from seatflow.worker.sse import make_tasks_router
from seatflow.worker.task_runner import TaskRunner
from seatflow.worker.task_store import (
    InMemoryTaskStore,
    InvalidStateTransitionError,
    TaskNotFoundError,
)


# ── Task store ──────────────────────────────────────────────────────────


class TestTaskStoreCRUD:
    def test_create_emits_created_event(self):
        store = InMemoryTaskStore()
        state = store.create(kind="experiment_run", spec={"a": 1})
        assert state.task_id
        assert state.status == TaskStatus.REQUESTED
        events = store.events(state.task_id)
        assert len(events) == 1
        assert events[0].event_type == "TASK_CREATED"

    def test_get_unknown_raises(self):
        store = InMemoryTaskStore()
        with pytest.raises(TaskNotFoundError):
            store.get("missing")

    def test_list_filtered_by_status(self):
        store = InMemoryTaskStore()
        a = store.create(kind="experiment_run", spec={})
        store.transition(a.task_id, TaskStatus.QUEUED)
        b = store.create(kind="backtest", spec={})
        # b stays REQUESTED.
        queued = store.list(status=TaskStatus.QUEUED)
        assert len(queued) == 1
        assert queued[0].task_id == a.task_id

    def test_list_filtered_by_kind(self):
        store = InMemoryTaskStore()
        store.create(kind="backtest", spec={})
        store.create(kind="prediction", spec={})
        bts = store.list(kind="backtest")
        assert len(bts) == 1

    def test_delete_removes_task_and_events(self):
        store = InMemoryTaskStore()
        state = store.create(kind="experiment_run", spec={})
        store.delete(state.task_id)
        with pytest.raises(TaskNotFoundError):
            store.get(state.task_id)
        assert store.events(state.task_id) == []

    def test_delete_unknown_raises(self):
        store = InMemoryTaskStore()
        with pytest.raises(TaskNotFoundError):
            store.delete("missing")


# ── State machine ───────────────────────────────────────────────────────


class TestTaskStoreTransitions:
    def test_requested_to_queued(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.QUEUED)
        assert store.get(s.task_id).status == TaskStatus.QUEUED

    def test_requested_to_running_skips_queued_is_allowed(self):
        # Per contract, REQUESTED may transition directly to RUNNING.
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        assert store.get(s.task_id).status == TaskStatus.RUNNING

    def test_running_to_completed(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        final = store.transition(
            s.task_id, TaskStatus.COMPLETED, progress=1.0, result={"ok": True}
        )
        assert final.status == TaskStatus.COMPLETED
        assert final.progress == 1.0
        assert final.result == {"ok": True}

    def test_terminal_completed_cannot_transition(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        store.transition(s.task_id, TaskStatus.COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            store.transition(s.task_id, TaskStatus.CANCELLED)

    def test_terminal_failed_cannot_transition(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        store.transition(s.task_id, TaskStatus.FAILED)
        with pytest.raises(InvalidStateTransitionError):
            store.transition(s.task_id, TaskStatus.RUNNING)

    def test_invalid_transition_raises(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        with pytest.raises(InvalidStateTransitionError):
            # REQUESTED -> COMPLETED is not allowed.
            store.transition(s.task_id, TaskStatus.COMPLETED)

    def test_progress_emits_event(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        store.progress(s.task_id, 0.5, message="halfway")
        event_types = [e.event_type for e in store.events(s.task_id)]
        assert "TASK_PROGRESS" in event_types
        progress_event = next(
            e for e in store.events(s.task_id) if e.event_type == "TASK_PROGRESS"
        )
        assert progress_event.payload["progress"] == 0.5
        assert progress_event.payload["message"] == "halfway"

    def test_progress_after_terminal_raises(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        store.transition(s.task_id, TaskStatus.COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            store.progress(s.task_id, 0.5)

    def test_cancel_queued_task(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.QUEUED)
        cancelled = store.cancel(s.task_id)
        assert cancelled.status == TaskStatus.CANCELLED

    def test_cancel_terminal_raises(self):
        store = InMemoryTaskStore()
        s = store.create(kind="x", spec={})
        store.transition(s.task_id, TaskStatus.RUNNING)
        store.transition(s.task_id, TaskStatus.COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            store.cancel(s.task_id)

    def test_is_terminal_helper(self):
        store = InMemoryTaskStore()
        assert store.is_terminal(TaskStatus.COMPLETED)
        assert store.is_terminal(TaskStatus.CANCELLED)
        assert store.is_terminal(TaskStatus.FAILED)
        assert not store.is_terminal(TaskStatus.REQUESTED)
        assert not store.is_terminal(TaskStatus.RUNNING)


# ── Task runner ─────────────────────────────────────────────────────────


class TestTaskRunner:
    async def test_submit_creates_queued_task(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        state = await runner.submit(kind="experiment_run", spec={"experiment_id": "e1"})
        assert state.status == TaskStatus.QUEUED
        assert state.task_id

    async def test_submit_unknown_kind_raises(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        with pytest.raises(ValueError):
            await runner.submit(kind="nope")

    async def test_run_once_drives_to_completion(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        state = await runner.submit(kind="experiment_run", spec={"experiment_id": "e1"})
        final = await runner.run_once()
        assert final is not None
        assert final.task_id == state.task_id
        assert final.status == TaskStatus.COMPLETED
        assert final.progress == 1.0
        assert final.result is not None
        assert final.result["experiment_id"] == "e1"

    async def test_run_once_emits_full_event_sequence(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        state = await runner.submit(kind="experiment_run", spec={})
        await runner.run_once()
        types = [e.event_type for e in store.events(state.task_id)]
        for required in (
            "TASK_CREATED",
            "TASK_QUEUED",
            "TASK_STARTED",
            "TASK_PROGRESS",
            "TASK_COMPLETED",
        ):
            assert required in types

    async def test_background_loop_drives_tasks(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        await runner.start()
        try:
            state = await runner.submit(kind="backtest", spec={})
            for _ in range(20):
                await asyncio.sleep(0)
                current = store.get(state.task_id)
                if current.status == TaskStatus.COMPLETED:
                    break
            assert store.get(state.task_id).status == TaskStatus.COMPLETED
        finally:
            await runner.stop()

    async def test_handler_exception_marks_task_failed(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)

        async def bad_handler(spec, runner_):
            raise RuntimeError("boom")

        runner.register_handler("bad", bad_handler)
        state = await runner.submit(kind="bad", spec={})
        final = await runner.run_once()
        assert final.status == TaskStatus.FAILED
        assert final.error is not None
        assert final.error.code == "TASK_FAILED"
        assert "boom" in final.error.message

    async def test_handler_missing_marks_task_failed(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        # Create a task whose kind has no registered handler.
        state = store.create(kind="orphan", spec={})
        store.transition(state.task_id, TaskStatus.QUEUED)
        # Push it back onto the runner's internal queue.
        await runner._enqueue(state.task_id)
        final = await runner.run_once()
        assert final.status == TaskStatus.FAILED

    async def test_register_handler_replaces_default(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)

        async def custom(spec, runner_):
            return {"custom": True}

        runner.register_handler("experiment_run", custom)
        state = await runner.submit(kind="experiment_run", spec={})
        final = await runner.run_once()
        assert final.result == {"custom": True}


# ── SSE endpoint ────────────────────────────────────────────────────────


class TestSSEEndpoint:
    async def test_sse_streams_events(self):
        store = InMemoryTaskStore()
        runner = TaskRunner(store)
        router = make_tasks_router(store)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            state = await runner.submit(kind="experiment_run", spec={})
            await runner.run_once()

            with client.stream("GET", f"/research/v1/tasks/{state.task_id}/events") as resp:
                assert resp.status_code == 200
                body = "".join(resp.iter_text())

        # The replayed event log must appear in the SSE body.
        assert "TASK_CREATED" in body
        assert "TASK_COMPLETED" in body
        # Each event is serialised as a JSON object on a ``data:`` line.
        assert "data: {" in body

    async def test_sse_unknown_task_returns_404(self):
        store = InMemoryTaskStore()
        router = make_tasks_router(store)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            resp = client.get("/research/v1/tasks/missing/events")
            assert resp.status_code == 404

    async def test_get_task_endpoint(self):
        store = InMemoryTaskStore()
        router = make_tasks_router(store)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            state = store.create(kind="x", spec={})
            resp = client.get(f"/research/v1/tasks/{state.task_id}")
            assert resp.status_code == 200
            assert resp.json()["taskId"] == state.task_id

    async def test_cancel_endpoint(self):
        store = InMemoryTaskStore()
        router = make_tasks_router(store)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            state = store.create(kind="x", spec={})
            store.transition(state.task_id, TaskStatus.QUEUED)
            resp = client.post(f"/research/v1/tasks/{state.task_id}/cancel")
            assert resp.status_code == 200
            assert resp.json()["status"] == "CANCELLED"

    async def test_cancel_unknown_returns_404(self):
        store = InMemoryTaskStore()
        router = make_tasks_router(store)
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            resp = client.post("/research/v1/tasks/missing/cancel")
            assert resp.status_code == 404
