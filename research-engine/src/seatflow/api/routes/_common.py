"""Shared route helpers for pagination and serialisation."""

from __future__ import annotations

from typing import Any, Sequence

from fastapi import Query

from seatflow.api.models import PaginationParams


def pagination_params(
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
) -> PaginationParams:
    """FastAPI dependency that parses pagination query parameters."""
    return PaginationParams(page=page, page_size=page_size)


def paginate(items: Sequence[Any], params: PaginationParams) -> tuple[list[Any], int]:
    """Slice *items* for the given page and return ``(page_items, total)``."""
    total = len(items)
    start = params.offset
    end = start + params.limit
    return list(items[start:end]), total
