"""FastAPI application factory for the SeatFlow Research Engine API.

Creates the application instance, registers all route modules, adds CORS
middleware, and installs a global exception handler that converts unhandled
errors into the standard ``ErrorResponse`` envelope.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from seatflow.api.models import ErrorResponse, HealthReport, HealthState
from seatflow.api.routes.backtests import router as backtests_router
from seatflow.api.routes.datasets import datasets_router, data_sources_router
from seatflow.api.routes.experiments import router as experiments_router
from seatflow.api.routes.models import router as models_router
from seatflow.api.routes.predictions import router as predictions_router
from seatflow.api.routes.projects import router as projects_router
from seatflow.api.routes.runs import router as runs_router
from seatflow.api.routes.state import _Store
from seatflow.domain.time import utc_now
from seatflow.worker.task_runner import ResearchPipelineAdapter, pipeline_capability

logger = logging.getLogger(__name__)


def create_app(
    *,
    reset_store: bool = False,
    fixture_mode: str | None = None,
    pipeline_adapter: ResearchPipelineAdapter | None = None,
) -> FastAPI:
    """Build and return the fully-configured FastAPI application.

    Parameters
    ----------
    reset_store:
        Retained for caller compatibility.  Every application factory now owns
        a fresh store, so state is isolated regardless of this flag.
    fixture_mode:
        Explicit non-production fixture mode.  The default remains empty.
    pipeline_adapter:
        Real research execution adapter used to report capability readiness.
    """
    app = FastAPI(
        title="SeatFlow Research Engine API",
        version="3.0.0-dev",
        description="Research Engine API for SeatFlow V3",
    )
    app_store = _Store(fixture_mode=fixture_mode)
    app.state.research_store = app_store

    # ── CORS ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────
    app.include_router(projects_router)
    app.include_router(data_sources_router)
    app.include_router(datasets_router)
    app.include_router(experiments_router)
    app.include_router(runs_router)
    app.include_router(models_router)
    app.include_router(predictions_router)
    app.include_router(backtests_router)

    # ── Health check ───────────────────────────────────────────────────
    @app.get(
        "/research/v1/health",
        response_model=HealthReport,
        tags=["health"],
        name="researchHealth",
    )
    def health_check() -> HealthReport:
        """Liveness/readiness probe for the research engine."""
        capability = pipeline_capability(pipeline_adapter)
        return HealthReport(
            status=(
                HealthState.UP
                if capability["status"] == HealthState.UP.value
                else HealthState.DEGRADED
            ),
            components={
                "process": {"status": HealthState.UP.value},
                "researchCapability": capability,
                "store": {
                    "status": HealthState.UP.value,
                    "mode": app_store.fixture_mode or "EMPTY_FAIL_CLOSED",
                },
            },
            timestamp=utc_now(),
        )

    # ── Global exception handler ───────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        trace_id = uuid.uuid4().hex
        route = request.scope.get("route")
        safe_path = getattr(route, "path", "/unknown")
        logger.exception(
            "Unhandled API exception trace_id=%s route=%s",
            trace_id,
            safe_path,
        )
        err = ErrorResponse.build(
            code="INTERNAL_ERROR",
            message="Internal server error",
            trace_id=trace_id,
            details={"path": safe_path},
            retryable=False,
            remediation="Contact the service operator with the traceId.",
        )
        return JSONResponse(
            status_code=500,
            content=err.model_dump(mode="json", by_alias=True),
        )

    return app


# Module-level default app instance (used by ``uvicorn seatflow.api.app:app``).
app = create_app()
