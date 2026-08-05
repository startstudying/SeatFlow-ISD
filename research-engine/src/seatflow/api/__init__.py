"""SeatFlow Research Engine API package.

Exposes :func:`create_app` and the default :data:`app` instance so the
package can be used directly by ``uvicorn``::

    uvicorn seatflow.api.app:app --reload
"""

from seatflow.api.app import app, create_app
from seatflow.api.models import (
    ErrorResponse,
    HealthReport,
    PaginatedResponse,
    PaginationParams,
)

__all__ = [
    "app",
    "create_app",
    "ErrorResponse",
    "HealthReport",
    "PaginatedResponse",
    "PaginationParams",
]
