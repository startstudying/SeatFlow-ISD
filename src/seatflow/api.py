from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

from seatflow.inference import InferenceContractError, InferenceQueryService
from seatflow.service import JsonDict, SnapshotNotFoundError, SnapshotQueryService
from seatflow.status import load_phase_status


def create_app(data_root: str | Path, provider: str = "sample") -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from starlette.responses import JSONResponse
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FastAPI is not installed. Install the optional API dependencies with "
            "`pip install 'seatflow[api]'`."
        ) from error

    service = SnapshotQueryService(data_root)
    inference_service = InferenceQueryService(data_root)
    app = FastAPI(
        title="SeatFlow API",
        version="0.1.0",
        description="Read-only HTTP layer over persisted SeatFlow snapshots.",
    )
    app.state.snapshot_service = service
    app.state.inference_service = inference_service
    app.state.default_provider = provider
    phase_state_path = Path(data_root).parent / "artifacts" / "status" / "phase-state.json"

    def inference_error_response(error: InferenceContractError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content=error.to_response())

    @app.get("/health", response_model=None)
    def health() -> JsonDict:
        return {
            "status": "ok",
            "provider": provider,
        }

    @app.get("/api/v1/audit/status", response_model=None)
    @app.get("/api/v1/status", response_model=None)
    def audit_status() -> JsonDict:
        return cast(JsonDict, load_phase_status(phase_state_path).to_dict())

    @app.get("/api/v1/providers/{provider_name}/snapshot-summary", response_model=None)
    def snapshot_summary(provider_name: str) -> JsonDict:
        try:
            return service.snapshot_summary(provider_name)
        except SnapshotNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/providers/{provider_name}/graph/{trade_date}", response_model=None)
    def graph_snapshot(provider_name: str, trade_date: date) -> JsonDict:
        try:
            return service.graph_snapshot(provider_name, trade_date)
        except SnapshotNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/v1/models/{model_id}", response_model=None)
    def model_status(model_id: str) -> JsonDict | JSONResponse:
        try:
            return inference_service.model_status(model_id)
        except InferenceContractError as error:
            return inference_error_response(error)

    @app.get("/api/v1/runs/{run_id}", response_model=None)
    def run_status(run_id: str) -> JsonDict | JSONResponse:
        try:
            return inference_service.run_status(run_id)
        except InferenceContractError as error:
            return inference_error_response(error)

    @app.get("/api/v1/backtests/{run_id}", response_model=None)
    def backtest_status(run_id: str) -> JsonDict | JSONResponse:
        try:
            return inference_service.backtest_status(run_id)
        except InferenceContractError as error:
            return inference_error_response(error)

    @app.post("/api/v1/providers/{provider_name}/models/{model_id}/predict", response_model=None)
    def predict(
        provider_name: str, model_id: str, request: dict[str, str] | None = None
    ) -> JsonDict | JSONResponse:
        try:
            trade_date = date.fromisoformat(cast(str, (request or {}).get("trade_date")))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="trade_date must use YYYY-MM-DD") from error
        target_stock_id = (request or {}).get("target_stock_id")
        if target_stock_id is None:
            raise HTTPException(status_code=422, detail="target_stock_id is required")
        try:
            return inference_service.prediction(
                provider=provider_name,
                model_id=model_id,
                trade_date=trade_date,
                target_stock_id=target_stock_id,
            )
        except InferenceContractError as error:
            return inference_error_response(error)

    return app
