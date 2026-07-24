from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_STATUS_KEYS = (
    "engineering_complete",
    "data_ready",
    "model_valid",
    "backtest_valid",
    "competition_ready",
)
_SCOPES = frozenset({"complete_seat", "institutional_only", "synthetic", "none"})
_DATA_SCOPES = frozenset({"real", "sample", "synthetic", "none"})


@dataclass(frozen=True)
class PhaseStatus:
    engineering_complete: bool
    data_ready: bool
    model_valid: bool
    backtest_valid: bool
    competition_ready: bool
    data_scope: str
    scope: str
    evidence_hashes: dict[str, str]
    blocked_reasons: tuple[str, ...]
    last_verified_at: str | None
    status_file_present: bool
    promotion_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "engineering_complete": self.engineering_complete,
            "data_ready": self.data_ready,
            "model_valid": self.model_valid,
            "backtest_valid": self.backtest_valid,
            "competition_ready": self.competition_ready,
            "data_scope": self.data_scope,
            "scope": self.scope,
            "seat_detail_scope": self.scope,
            "evidence_hashes": dict(self.evidence_hashes),
            "blocked_reasons": list(self.blocked_reasons),
            "last_verified_at": self.last_verified_at,
            "status_file_present": self.status_file_present,
        }
        if self.promotion_decision is not None:
            payload["promotion_decision"] = dict(self.promotion_decision)
        return payload


def default_phase_status() -> PhaseStatus:
    return PhaseStatus(
        engineering_complete=False,
        data_ready=False,
        model_valid=False,
        backtest_valid=False,
        competition_ready=False,
        data_scope="none",
        scope="none",
        evidence_hashes={},
        blocked_reasons=("No verified phase-state artifact exists.",),
        last_verified_at=None,
        status_file_present=False,
        promotion_decision=None,
    )


def load_phase_status(path: str | Path) -> PhaseStatus:
    status_path = Path(path)
    if not status_path.exists():
        return default_phase_status()
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _blocked_status(f"Phase-state artifact is unreadable: {type(error).__name__}")
    if not isinstance(raw, dict):
        return _blocked_status("Phase-state artifact must be a JSON object")
    try:
        return _from_dict(raw, present=True)
    except ValueError as error:
        return _blocked_status(f"Phase-state artifact is invalid: {error}")


def write_phase_status(path: str | Path, status: PhaseStatus) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(status.to_dict(), ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    with NamedTemporaryFile(
        "wb", delete=False, dir=output_path.parent, prefix=".phase-state."
    ) as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    return output_path


def new_phase_status(
    *,
    engineering_complete: bool,
    data_ready: bool,
    model_valid: bool,
    backtest_valid: bool,
    competition_ready: bool,
    scope: str,
    evidence_hashes: dict[str, str],
    blocked_reasons: tuple[str, ...] = (),
    data_scope: str | None = None,
) -> PhaseStatus:
    raw: dict[str, Any] = {
        "engineering_complete": engineering_complete,
        "data_ready": data_ready,
        "model_valid": model_valid,
        "backtest_valid": backtest_valid,
        "competition_ready": competition_ready,
        "data_scope": data_scope or _derive_data_scope(scope, data_ready),
        "scope": scope,
        "evidence_hashes": evidence_hashes,
        "blocked_reasons": list(blocked_reasons),
        "last_verified_at": datetime.now().astimezone().isoformat(),
    }
    return _from_dict(raw, present=True)


def phase_status_from_promotion(
    *,
    engineering_complete: bool,
    data_ready: bool,
    scope: str,
    evidence_hashes: dict[str, str],
    promotion_decision: Any,
    data_scope: str | None = None,
) -> PhaseStatus:
    decision_payload = (
        promotion_decision.to_dict()
        if hasattr(promotion_decision, "to_dict")
        else dict(promotion_decision)
    )
    blocked_reasons = list(decision_payload.get("blocked_reasons", []))
    consistency_reasons = _promotion_consistency_reasons(decision_payload)
    blocked_reasons.extend(consistency_reasons)
    promoted = decision_payload.get("promoted") is True and not consistency_reasons
    model_valid = promoted and data_ready and decision_payload.get("model_valid") is True
    backtest_valid = model_valid and decision_payload.get("backtest_valid") is True
    competition_ready = backtest_valid and decision_payload.get("competition_ready") is True
    if promoted and not data_ready:
        blocked_reasons.append("data_ready_required_for_validity_claims")
    raw: dict[str, Any] = {
        "engineering_complete": engineering_complete,
        "data_ready": data_ready,
        "model_valid": model_valid,
        "backtest_valid": backtest_valid,
        "competition_ready": competition_ready,
        "data_scope": data_scope or _derive_data_scope(scope, data_ready),
        "scope": scope,
        "evidence_hashes": evidence_hashes,
        "blocked_reasons": blocked_reasons,
        "last_verified_at": datetime.now().astimezone().isoformat(),
        "promotion_decision": decision_payload,
    }
    return _from_dict(raw, present=True)


def _from_dict(raw: dict[str, Any], *, present: bool) -> PhaseStatus:
    values: dict[str, bool] = {}
    for key in _STATUS_KEYS:
        value = raw.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        values[key] = value
    scope = raw.get("seat_detail_scope", raw.get("scope"))
    if not isinstance(scope, str) or scope not in _SCOPES:
        raise ValueError(f"scope must be one of {sorted(_SCOPES)}")
    data_scope = raw.get("data_scope", "none")
    if not isinstance(data_scope, str) or data_scope not in _DATA_SCOPES:
        raise ValueError(f"data_scope must be one of {sorted(_DATA_SCOPES)}")
    evidence_hashes = raw.get("evidence_hashes", {})
    if not isinstance(evidence_hashes, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in evidence_hashes.items()
    ):
        raise ValueError("evidence_hashes must be a string map")
    blocked_reasons = raw.get("blocked_reasons", [])
    if not isinstance(blocked_reasons, list) or not all(
        isinstance(reason, str) for reason in blocked_reasons
    ):
        raise ValueError("blocked_reasons must be a string list")
    last_verified_at = raw.get("last_verified_at")
    if last_verified_at is not None and not isinstance(last_verified_at, str):
        raise ValueError("last_verified_at must be a string or null")
    promotion_decision = raw.get("promotion_decision")
    if promotion_decision is not None and not isinstance(promotion_decision, dict):
        raise ValueError("promotion_decision must be an object")
    if promotion_decision is not None and promotion_decision.get("promoted") is not True and (
        values["model_valid"] or values["backtest_valid"] or values["competition_ready"]
    ):
        raise ValueError("validity claims require promoted promotion_decision")
    if promotion_decision is not None:
        if values["model_valid"] and promotion_decision.get("model_valid") is not True:
            raise ValueError("model_valid requires promotion_decision.model_valid=true")
        if values["backtest_valid"] and promotion_decision.get("backtest_valid") is not True:
            raise ValueError("backtest_valid requires promotion_decision.backtest_valid=true")
        if values["competition_ready"] and promotion_decision.get("competition_ready") is not True:
            raise ValueError("competition_ready requires promotion_decision.competition_ready=true")
    if values["model_valid"] and not values["data_ready"]:
        raise ValueError("model_valid requires data_ready")
    if values["backtest_valid"] and not values["model_valid"]:
        raise ValueError("backtest_valid requires model_valid")
    if values["competition_ready"] and not values["backtest_valid"]:
        raise ValueError("competition_ready requires backtest_valid")
    if values["data_ready"] and data_scope != "real":
        raise ValueError("data_ready requires data_scope=real")
    if scope in {"none", "synthetic"} and (
        values["data_ready"] or values["model_valid"] or values["backtest_valid"]
    ):
        raise ValueError("synthetic/none scope cannot claim data, model, or backtest validity")
    return PhaseStatus(
        engineering_complete=values["engineering_complete"],
        data_ready=values["data_ready"],
        model_valid=values["model_valid"],
        backtest_valid=values["backtest_valid"],
        competition_ready=values["competition_ready"],
        data_scope=data_scope,
        scope=scope,
        evidence_hashes=dict(evidence_hashes),
        blocked_reasons=tuple(blocked_reasons),
        last_verified_at=last_verified_at,
        status_file_present=present,
        promotion_decision=dict(promotion_decision) if promotion_decision is not None else None,
    )


def _blocked_status(reason: str) -> PhaseStatus:
    default = default_phase_status()
    return PhaseStatus(
        engineering_complete=False,
        data_ready=False,
        model_valid=False,
        backtest_valid=False,
        competition_ready=False,
        data_scope="none",
        scope="none",
        evidence_hashes={},
        blocked_reasons=(reason, *default.blocked_reasons),
        last_verified_at=None,
        status_file_present=True,
        promotion_decision=None,
    )


def _derive_data_scope(scope: str, data_ready: bool) -> str:
    if data_ready and scope in {"complete_seat", "institutional_only"}:
        return "real"
    if scope == "synthetic":
        return "synthetic"
    return "none"


def _promotion_consistency_reasons(promotion_decision: dict[str, Any]) -> tuple[str, ...]:
    promoted = promotion_decision.get("promoted") is True
    model_valid = promotion_decision.get("model_valid") is True
    backtest_valid = promotion_decision.get("backtest_valid") is True
    competition_ready = promotion_decision.get("competition_ready") is True
    reasons: list[str] = []
    if promoted and not model_valid:
        reasons.append("promotion_model_valid_not_true")
    if promoted and not backtest_valid:
        reasons.append("promotion_backtest_valid_not_true")
    if model_valid and not promoted:
        reasons.append("promotion_model_valid_without_promoted")
    if backtest_valid and not model_valid:
        reasons.append("promotion_backtest_valid_without_model_valid")
    if competition_ready and not backtest_valid:
        reasons.append("promotion_competition_ready_without_backtest_valid")
    return tuple(reasons)
