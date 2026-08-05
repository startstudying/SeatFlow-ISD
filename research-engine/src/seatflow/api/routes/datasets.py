"""Dataset and data-source routes.

* ``/research/v1/datasets``
* ``/research/v1/data-sources``
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    DataSource,
    DataSourceKind,
    Dataset,
    DatasetStatus,
    PaginatedResponse,
    PaginationParams,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import store

# ── Data sources ────────────────────────────────────────────────────────

data_sources_router = APIRouter(
    prefix="/research/v1/data-sources", tags=["data-sources"]
)


@data_sources_router.get(
    "", response_model=PaginatedResponse[DataSource], name="list_data_sources"
)
def list_data_sources(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    kind: str | None = Query(None, description="Filter by data source kind"),
    enabled: bool | None = Query(None, description="Filter by enabled state"),
) -> PaginatedResponse[DataSource]:
    """List data sources with pagination and optional filters."""
    items = list(store.data_sources.values())
    if kind is not None:
        items = [d for d in items if d.kind == DataSourceKind(kind)]
    if enabled is not None:
        items = [d for d in items if d.enabled == enabled]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[DataSource].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@data_sources_router.get(
    "/{source_id}", response_model=DataSource, name="get_data_source"
)
def get_data_source(source_id: str) -> DataSource:
    """Retrieve a single data source by id."""
    source = store.data_sources.get(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Data source '{source_id}' not found",
            },
        )
    return source


# ── Datasets ────────────────────────────────────────────────────────────

datasets_router = APIRouter(prefix="/research/v1/datasets", tags=["datasets"])


@datasets_router.get(
    "", response_model=PaginatedResponse[Dataset], name="list_datasets"
)
def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description="Filter by dataset status"),
    project_id: str | None = Query(None, description="Filter by project id"),
) -> PaginatedResponse[Dataset]:
    """List datasets with pagination and optional filters."""
    items = list(store.datasets.values())
    if status is not None:
        items = [d for d in items if d.status == DatasetStatus(status)]
    if project_id is not None:
        items = [d for d in items if d.project_id == project_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Dataset].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@datasets_router.get("/{dataset_id}", response_model=Dataset, name="get_dataset")
def get_dataset(dataset_id: str) -> Dataset:
    """Retrieve a single dataset by id."""
    dataset = store.datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Dataset '{dataset_id}' not found"},
        )
    return dataset
