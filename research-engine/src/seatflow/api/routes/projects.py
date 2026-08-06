"""Project routes — ``/research/v1/projects``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    PaginatedResponse,
    PaginationParams,
    Project,
    ProjectCreate,
    ProjectStatus,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import StoreDependency, _gen_id
from seatflow.domain.time import utc_now

router = APIRouter(prefix="/research/v1/projects", tags=["projects"])


@router.get("", response_model=PaginatedResponse[Project], name="list_projects")
def list_projects(
    store: StoreDependency,
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    status: str | None = Query(None, description="Filter by project status"),
    owner: str | None = Query(None, description="Filter by owner"),
) -> PaginatedResponse[Project]:
    """List projects with pagination.

    Optional ``status`` and ``owner`` query filters are supported.
    """
    items = list(store.projects.values())
    if status is not None:
        items = [p for p in items if p.status == ProjectStatus(status)]
    if owner is not None:
        items = [p for p in items if p.owner == owner]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Project].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.post(
    "",
    response_model=Project,
    status_code=status.HTTP_201_CREATED,
    name="create_project",
)
def create_project(body: ProjectCreate, store: StoreDependency) -> Project:
    """Create a new project."""
    now = utc_now()
    project = Project(
        id=_gen_id(),
        name=body.name,
        description=body.description,
        owner=body.owner,
        status=ProjectStatus.DRAFT,
        tags=list(body.tags),
        created_at=now,
        updated_at=now,
    )
    store.projects[project.id] = project
    return project


@router.get("/{project_id}", response_model=Project, name="get_project")
def get_project(project_id: str, store: StoreDependency) -> Project:
    """Retrieve a single project by id."""
    project = store.projects.get(project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Project '{project_id}' not found"},
        )
    return project
