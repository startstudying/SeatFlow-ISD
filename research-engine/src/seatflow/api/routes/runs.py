"""Run routes — ``/research/v1/runs``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    PaginatedResponse,
    PaginationParams,
    Run,
    RunCreate,
    RunStatus,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import _gen_id, store
from seatflow.domain.time import utc_now

router = APIRouter(prefix="/research/v1/runs", tags=["runs"])


@router.get("", response_model=PaginatedResponse[Run], name="list_runs")
def list_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description="Filter by run status"),
    experiment_id: str | None = Query(None, description="Filter by experiment id"),
) -> PaginatedResponse[Run]:
    """List runs with pagination and optional filters."""
    items = list(store.runs.values())
    if status is not None:
        items = [r for r in items if r.status == RunStatus(status)]
    if experiment_id is not None:
        items = [r for r in items if r.experiment_id == experiment_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Run].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.post(
    "",
    response_model=Run,
    status_code=status.HTTP_201_CREATED,
    name="create_run",
)
def create_run(body: RunCreate) -> Run:
    """Create a new run for an experiment.

    Returns 404 if the referenced experiment does not exist.
    """
    if body.experiment_id not in store.experiments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Experiment '{body.experiment_id}' not found",
            },
        )
    now = utc_now()
    run = Run(
        id=_gen_id(),
        experiment_id=body.experiment_id,
        status=RunStatus.PENDING,
        progress=0.0,
        params=dict(body.params),
        created_at=now,
    )
    store.runs[run.id] = run
    return run


@router.get("/{run_id}", response_model=Run, name="get_run")
def get_run(run_id: str) -> Run:
    """Retrieve a single run by id."""
    run = store.runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Run '{run_id}' not found"},
        )
    return run
