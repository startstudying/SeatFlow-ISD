"""Prediction routes — ``/research/v1/predictions``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import PaginatedResponse, PaginationParams, Prediction
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import store

router = APIRouter(prefix="/research/v1/predictions", tags=["predictions"])


@router.get(
    "", response_model=PaginatedResponse[Prediction], name="list_predictions"
)
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    model_id: str | None = Query(None, description="Filter by model id"),
) -> PaginatedResponse[Prediction]:
    """List predictions with pagination and optional filters."""
    items = list(store.predictions.values())
    if model_id is not None:
        items = [p for p in items if p.model_id == model_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Prediction].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.get(
    "/{prediction_id}", response_model=Prediction, name="get_prediction"
)
def get_prediction(prediction_id: str) -> Prediction:
    """Retrieve a single prediction by id."""
    pred = store.predictions.get(prediction_id)
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Prediction '{prediction_id}' not found",
            },
        )
    return pred
