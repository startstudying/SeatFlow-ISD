"""Backtest routes — ``/research/v1/backtests``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from seatflow.api.models import (
    Backtest,
    BacktestCreate,
    BacktestStatus,
    PaginatedResponse,
    PaginationParams,
)
from seatflow.api.routes._common import paginate
from seatflow.api.routes.state import _gen_id, store
from seatflow.domain.time import utc_now

router = APIRouter(prefix="/research/v1/backtests", tags=["backtests"])


@router.get(
    "", response_model=PaginatedResponse[Backtest], name="list_backtests"
)
def list_backtests(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: str | None = Query(None, description="Filter by backtest status"),
    model_id: str | None = Query(None, description="Filter by model id"),
) -> PaginatedResponse[Backtest]:
    """List backtests with pagination and optional filters."""
    items = list(store.backtests.values())
    if status is not None:
        items = [b for b in items if b.status == BacktestStatus(status)]
    if model_id is not None:
        items = [b for b in items if b.model_id == model_id]
    params = PaginationParams(page=page, page_size=page_size)
    page_items, total = paginate(items, params)
    return PaginatedResponse[Backtest].from_items(
        page_items, page=page, page_size=page_size, total=total
    )


@router.post(
    "",
    response_model=Backtest,
    status_code=status.HTTP_201_CREATED,
    name="create_backtest",
)
def create_backtest(body: BacktestCreate) -> Backtest:
    """Create a new backtest definition.

    Returns 404 if the referenced model or dataset does not exist.
    """
    if body.model_id not in store.models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Model '{body.model_id}' not found",
            },
        )
    if body.dataset_id not in store.datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Dataset '{body.dataset_id}' not found",
            },
        )
    now = utc_now()
    bt = Backtest(
        id=_gen_id(),
        name=body.name,
        model_id=body.model_id,
        dataset_id=body.dataset_id,
        status=BacktestStatus.PENDING,
        range_from=body.range_from,
        range_to=body.range_to,
        created_at=now,
    )
    store.backtests[bt.id] = bt
    return bt


@router.get("/{backtest_id}", response_model=Backtest, name="get_backtest")
def get_backtest(backtest_id: str) -> Backtest:
    """Retrieve a single backtest by id."""
    bt = store.backtests.get(backtest_id)
    if bt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"Backtest '{backtest_id}' not found",
            },
        )
    return bt
