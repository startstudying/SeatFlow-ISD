"""Model routes — ``/research/v1/models``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    ModelInfo,
    ModelKind,
    PaginatedResponse,
    PaginationParams,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import store

router = APIRouter(prefix="/research/v1/models", tags=["models"])


@router.get("", response_model=PaginatedResponse[ModelInfo], name="list_models")
def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    kind: str | None = Query(None, description="Filter by model kind"),
    experiment_id: str | None = Query(None, description="Filter by experiment id"),
) -> PaginatedResponse[ModelInfo]:
    """List models with pagination and optional filters."""
    items = list(store.models.values())
    if kind is not None:
        items = [m for m in items if m.kind == ModelKind(kind)]
    if experiment_id is not None:
        items = [m for m in items if m.experiment_id == experiment_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[ModelInfo].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.get("/{model_id}", response_model=ModelInfo, name="get_model")
def get_model(model_id: str) -> ModelInfo:
    """Retrieve a single model by id."""
    model = store.models.get(model_id)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Model '{model_id}' not found"},
        )
    return model
