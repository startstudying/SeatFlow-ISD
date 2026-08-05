"""Experiment routes — ``/research/v1/experiments``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
    PaginatedResponse,
    PaginationParams,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import _gen_id, store
from seatflow.domain.time import utc_now

router = APIRouter(prefix="/research/v1/experiments", tags=["experiments"])


@router.get(
    "", response_model=PaginatedResponse[Experiment], name="list_experiments"
)
def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description="Filter by experiment status"),
    project_id: str | None = Query(None, description="Filter by project id"),
) -> PaginatedResponse[Experiment]:
    """List experiments with pagination and optional filters."""
    items = list(store.experiments.values())
    if status is not None:
        items = [e for e in items if e.status == ExperimentStatus(status)]
    if project_id is not None:
        items = [e for e in items if e.project_id == project_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Experiment].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.post(
    "",
    response_model=Experiment,
    status_code=status.HTTP_201_CREATED,
    name="create_experiment",
)
def create_experiment(body: ExperimentCreate) -> Experiment:
    """Create a new experiment."""
    now = utc_now()
    experiment = Experiment(
        id=_gen_id(),
        name=body.name,
        project_id=body.project_id,
        description=body.description,
        status=ExperimentStatus.DRAFT,
        config=dict(body.config),
        created_at=now,
        updated_at=now,
    )
    store.experiments[experiment.id] = experiment
    return experiment


@router.get(
    "/{experiment_id}", response_model=Experiment, name="get_experiment"
)
def get_experiment(experiment_id: str) -> Experiment:
    """Retrieve a single experiment by id."""
    experiment = store.experiments.get(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Experiment '{experiment_id}' not found",
            },
        )
    return experiment
