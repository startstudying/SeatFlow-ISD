from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

from seatflow.service import JsonDict, JsonValue, SnapshotNotFoundError, SnapshotQueryService


class InferenceContractError(LookupError):
    """Base error carrying a fail-closed inference/API response payload."""

    status_code = 409

    def __init__(
        self,
        message: str,
        *,
        model_validity_status: str = "unavailable",
        data_scope: str = "unknown",
        seat_detail_scope: str = "unknown",
        lineage_hash: str | None = None,
        risk_flags: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.model_validity_status = model_validity_status
        self.data_scope = data_scope
        self.seat_detail_scope = seat_detail_scope
        self.lineage_hash = lineage_hash
        self.risk_flags = risk_flags or ("inference_contract_unavailable",)

    def to_response(self) -> JsonDict:
        return {
            "error": self.message,
            "model_validity_status": self.model_validity_status,
            "data_scope": self.data_scope,
            "seat_detail_scope": self.seat_detail_scope,
            "lineage_hash": self.lineage_hash,
            "risk_flags": list(self.risk_flags),
        }


class InferenceNotFoundError(InferenceContractError):
    status_code = 404


class InferenceConflictError(InferenceContractError):
    status_code = 409


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    run_id: str
    checkpoint_path: Path
    model_validity_status: str
    data_scope: str
    seat_detail_scope: str
    lineage_hash: str
    risk_flags: tuple[str, ...]
    metadata: JsonDict

    def contract_fields(self) -> JsonDict:
        return {
            "model_validity_status": self.model_validity_status,
            "data_scope": self.data_scope,
            "seat_detail_scope": self.seat_detail_scope,
            "lineage_hash": self.lineage_hash,
            "risk_flags": list(self.risk_flags),
        }


class InferenceQueryService:
    """Fail-closed facade over persisted model, lineage, backtest, and prediction artifacts."""

    _IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.snapshot_service = SnapshotQueryService(self.data_root)

    def model_status(self, model_id: str) -> JsonDict:
        model = self._require_servable_model(model_id)
        lineage = self._require_lineage(model.lineage_hash, model)
        return {
            "model_id": model.model_id,
            "run_id": model.run_id,
            "checkpoint_path": str(model.checkpoint_path),
            "metadata": model.metadata,
            "lineage": lineage,
            **model.contract_fields(),
        }

    def run_status(self, run_id: str) -> JsonDict:
        self._validate_identifier(run_id, "run_id")
        run_path = self._find_existing(self._artifact_candidates(f"runs/{run_id}.json"))
        if run_path is None:
            raise InferenceNotFoundError(
                f"run artifact not found for run '{run_id}'",
                risk_flags=("run_artifact_missing",),
            )
        run = self._read_json_dict(run_path)
        return self._validated_contract_payload(run, f"run artifact is not servable: {run_path}")

    def backtest_status(self, run_id: str) -> JsonDict:
        self._validate_identifier(run_id, "run_id")
        backtest_path = self._find_existing(self._artifact_candidates(f"backtests/{run_id}.json"))
        if backtest_path is None:
            raise InferenceNotFoundError(
                f"backtest artifact not found for run '{run_id}'",
                risk_flags=("backtest_artifact_missing",),
            )
        backtest = self._read_json_dict(backtest_path)
        return self._validated_contract_payload(
            backtest, f"backtest artifact is not servable: {backtest_path}"
        )

    def prediction(
        self,
        *,
        provider: str,
        model_id: str,
        trade_date: date,
        target_stock_id: str,
    ) -> JsonDict:
        model = self._require_servable_model(model_id)
        lineage = self._require_lineage(model.lineage_hash, model)
        graph = self._require_real_graph(provider, trade_date, model)
        prediction = self._require_prediction_artifact(
            model_id=model.model_id,
            provider=provider,
            trade_date=trade_date,
            target_stock_id=target_stock_id,
            model=model,
        )
        return {
            "provider": provider,
            "model_id": model.model_id,
            "run_id": model.run_id,
            "trade_date": trade_date.isoformat(),
            "target_stock_id": target_stock_id,
            "prediction": prediction,
            "lineage": lineage,
            "graph_manifest": graph["manifest"],
            **model.contract_fields(),
        }

    def _require_servable_model(self, model_id: str) -> ModelArtifact:
        self._validate_identifier(model_id, "model_id")
        metadata_path = self._find_existing(
            self._checkpoint_candidates(f"{model_id}/metadata.json")
            + self._checkpoint_candidates(f"{model_id}.json")
        )
        if metadata_path is None:
            raise InferenceNotFoundError(
                f"checkpoint metadata not found for model '{model_id}'",
                risk_flags=("checkpoint_missing",),
            )
        metadata = self._read_json_dict(metadata_path)
        checkpoint_name = self._required_str(metadata, "checkpoint_file", "checkpoint missing")
        checkpoint_path = metadata_path.parent / checkpoint_name
        if not checkpoint_path.exists():
            raise InferenceNotFoundError(
                f"checkpoint file not found for model '{model_id}'",
                model_validity_status=self._str_value(metadata.get("model_validity_status")),
                data_scope=self._str_value(metadata.get("data_scope")),
                lineage_hash=self._optional_str(metadata.get("lineage_hash")),
                risk_flags=("checkpoint_file_missing",),
            )

        model = ModelArtifact(
            model_id=self._required_str(metadata, "model_id", "model metadata missing model_id"),
            run_id=self._required_str(metadata, "run_id", "model metadata missing run_id"),
            checkpoint_path=checkpoint_path,
            model_validity_status=self._required_str(
                metadata, "model_validity_status", "model metadata missing validity status"
            ),
            data_scope=self._required_str(
                metadata, "data_scope", "model metadata missing data_scope"
            ),
            seat_detail_scope=self._required_seat_detail_scope(
                metadata, "model metadata missing seat_detail_scope"
            ),
            lineage_hash=self._required_str(
                metadata, "lineage_hash", "model metadata missing lineage_hash"
            ),
            risk_flags=self._string_tuple(metadata.get("risk_flags")),
            metadata=metadata,
        )
        if model.model_id != model_id:
            raise InferenceConflictError(
                f"model metadata id mismatch for model '{model_id}'",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("model_id_mismatch",),
            )
        if model.model_validity_status != "valid":
            raise InferenceConflictError(
                f"model '{model_id}' is not marked valid",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("model_not_valid",),
            )
        if model.data_scope != "real":
            raise InferenceConflictError(
                f"model '{model_id}' was not trained on real data",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("non_real_model_data_scope",),
            )
        return model

    def _require_lineage(self, lineage_hash: str, model: ModelArtifact) -> JsonDict:
        lineage_path = self._find_existing(
            self._artifact_candidates(f"lineage/{lineage_hash}.json")
        )
        if lineage_path is None:
            raise InferenceConflictError(
                f"lineage artifact not found for hash '{lineage_hash}'",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("lineage_missing",),
            )
        lineage = self._read_json_dict(lineage_path)
        payload = self._validated_contract_payload(
            lineage, f"lineage artifact is not servable: {lineage_path}"
        )
        if payload["lineage_hash"] != lineage_hash:
            raise InferenceConflictError(
                f"lineage artifact hash mismatch for hash '{lineage_hash}'",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("lineage_hash_mismatch",),
            )
        if payload["seat_detail_scope"] != model.seat_detail_scope:
            raise InferenceConflictError(
                "lineage seat_detail_scope does not match model metadata",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("lineage_seat_detail_scope_mismatch",),
            )
        return payload

    def _require_real_graph(
        self, provider: str, trade_date: date, model: ModelArtifact
    ) -> JsonDict:
        try:
            graph = self.snapshot_service.graph_snapshot(provider, trade_date)
        except SnapshotNotFoundError as error:
            raise InferenceNotFoundError(
                str(error),
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("graph_snapshot_missing",),
            ) from error

        manifest = graph.get("manifest")
        metadata: JsonDict = {}
        if isinstance(manifest, dict):
            manifest_metadata = manifest.get("metadata")
            if isinstance(manifest_metadata, dict):
                metadata = manifest_metadata
        if metadata.get("data_scope") != "real":
            raise InferenceConflictError(
                "graph snapshot for provider "
                f"'{provider}' on {trade_date.isoformat()} is not real data",
                model_validity_status=model.model_validity_status,
                data_scope=self._str_value(metadata.get("data_scope")),
                seat_detail_scope=self._str_value(metadata.get("seat_detail_scope")),
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("non_real_graph_data_scope",),
            )
        graph_scope = self._required_seat_detail_scope(
            metadata, "graph metadata missing seat_detail_scope"
        )
        if graph_scope != model.seat_detail_scope:
            raise InferenceConflictError(
                "graph seat_detail_scope does not match model metadata",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("graph_seat_detail_scope_mismatch",),
            )
        return graph

    def _require_prediction_artifact(
        self,
        *,
        model_id: str,
        provider: str,
        trade_date: date,
        target_stock_id: str,
        model: ModelArtifact,
    ) -> JsonDict:
        self._validate_identifier(provider, "provider")
        self._validate_prediction_key(target_stock_id)
        relative = (
            f"predictions/{model_id}/{provider}/{trade_date.isoformat()}/{target_stock_id}.json"
        )
        prediction_path = self._find_existing(self._artifact_candidates(relative))
        if prediction_path is None:
            raise InferenceNotFoundError(
                "prediction artifact not found for model "
                f"'{model_id}' and stock '{target_stock_id}'",
                model_validity_status=model.model_validity_status,
                data_scope=model.data_scope,
                seat_detail_scope=model.seat_detail_scope,
                lineage_hash=model.lineage_hash,
                risk_flags=model.risk_flags + ("prediction_artifact_missing",),
            )
        prediction = self._read_json_dict(prediction_path)
        payload = self._validated_contract_payload(
            prediction, f"prediction artifact is not servable: {prediction_path}"
        )
        expected_fields = {
            "lineage_hash": model.lineage_hash,
            "model_id": model.model_id,
            "provider": provider,
            "trade_date": trade_date.isoformat(),
            "target_stock_id": target_stock_id,
            "seat_detail_scope": model.seat_detail_scope,
        }
        for field, expected in expected_fields.items():
            actual = self._required_str(
                payload, field, f"prediction artifact missing binding field: {field}"
            )
            if actual != expected:
                raise InferenceConflictError(
                    f"prediction artifact {field} does not match the requested model context",
                    model_validity_status=model.model_validity_status,
                    data_scope=model.data_scope,
                    seat_detail_scope=model.seat_detail_scope,
                    lineage_hash=model.lineage_hash,
                    risk_flags=model.risk_flags + (f"prediction_{field}_mismatch",),
                )
        return payload

    def _validated_contract_payload(self, payload: JsonDict, message: str) -> JsonDict:
        model_validity_status = self._required_str(
            payload, "model_validity_status", f"{message}; missing model_validity_status"
        )
        data_scope = self._required_str(payload, "data_scope", f"{message}; missing data_scope")
        seat_detail_scope = self._required_seat_detail_scope(
            payload, f"{message}; missing seat_detail_scope"
        )
        lineage_hash = self._required_str(
            payload, "lineage_hash", f"{message}; missing lineage_hash"
        )
        risk_flags = self._string_tuple(payload.get("risk_flags"))
        if model_validity_status != "valid":
            raise InferenceConflictError(
                message,
                model_validity_status=model_validity_status,
                data_scope=data_scope,
                seat_detail_scope=seat_detail_scope,
                lineage_hash=lineage_hash,
                risk_flags=risk_flags + ("artifact_model_not_valid",),
            )
        if data_scope != "real":
            raise InferenceConflictError(
                message,
                model_validity_status=model_validity_status,
                data_scope=data_scope,
                seat_detail_scope=seat_detail_scope,
                lineage_hash=lineage_hash,
                risk_flags=risk_flags + ("artifact_non_real_data_scope",),
            )
        response = dict(payload)
        response["risk_flags"] = list(risk_flags)
        return response

    def _checkpoint_candidates(self, relative_path: str) -> list[Path]:
        return [
            self.data_root / "checkpoints" / relative_path,
            self.data_root.parent / "checkpoints" / relative_path,
        ]

    def _artifact_candidates(self, relative_path: str) -> list[Path]:
        return [
            self.data_root / "artifacts" / relative_path,
            self.data_root.parent / "artifacts" / relative_path,
        ]

    @staticmethod
    def _find_existing(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None

    @staticmethod
    def _read_json_dict(path: Path) -> JsonDict:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"inference artifact must be an object: {path}")
        return cast(JsonDict, raw)

    @classmethod
    def _validate_identifier(cls, value: str, name: str) -> None:
        if not cls._IDENTIFIER_PATTERN.fullmatch(value):
            raise InferenceNotFoundError(
                f"{name} identifier is invalid", risk_flags=("invalid_id",)
            )

    @staticmethod
    def _validate_prediction_key(value: str) -> None:
        if "/" in value or "\\" in value or not value.strip():
            raise InferenceNotFoundError(
                "target_stock_id is invalid", risk_flags=("invalid_stock_id",)
            )

    @staticmethod
    def _required_str(payload: JsonDict, key: str, message: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise InferenceConflictError(message, risk_flags=(f"{key}_missing",))
        return value

    @staticmethod
    def _optional_str(value: JsonValue) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _required_seat_detail_scope(payload: JsonDict, message: str) -> str:
        scope = InferenceQueryService._required_str(payload, "seat_detail_scope", message)
        if scope not in {"complete_seat", "institutional_only"}:
            raise InferenceConflictError(
                "seat_detail_scope is not servable",
                seat_detail_scope=scope,
                risk_flags=("seat_detail_scope_not_servable",),
            )
        return scope

    @staticmethod
    def _str_value(value: JsonValue) -> str:
        return value if isinstance(value, str) and value else "unknown"

    @staticmethod
    def _string_tuple(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str))
