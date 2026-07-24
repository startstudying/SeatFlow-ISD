"""Compose authoritative M0 data-contract evidence from fail-closed audits."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def main() -> None:
    args = _arguments()
    identity = _load_audit(Path(args.identity_audit), "identity")
    labels = _load_audit(Path(args.label_audit), "label")
    protocol = _load_protocol(Path(args.protocol_lock))
    payload = _compose(identity, labels, protocol)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "blocked": len(payload["blocked_reasons"])}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-audit",
        default="artifacts/diagnostics/M0_SEAT_IDENTITY_STABILITY_AUDIT/result.json",
    )
    parser.add_argument(
        "--label-audit",
        default="artifacts/diagnostics/R0_V2_LABEL_COVERAGE_AUDIT/result.json",
    )
    parser.add_argument(
        "--protocol-lock",
        default="artifacts/protocol/seatflow-production-v1-v2/protocol.lock.json",
    )
    parser.add_argument(
        "--output", default="artifacts/diagnostics/M0_DATA_CONTRACT_AUDIT/result.json"
    )
    return parser.parse_args()


def _load_audit(path: Path, kind: str) -> dict[str, object]:
    raw = _load_object(path)
    if raw.get("status") != "audited_fail_closed" or raw.get("model_valid") is not False:
        raise ValueError(f"{kind} audit must be fail-closed: {path}")
    raw["_path"] = str(path)
    raw["_file_sha256"] = sha256(path.read_bytes()).hexdigest()
    return raw


def _load_protocol(path: Path) -> dict[str, object]:
    raw = _load_object(path)
    claims = raw.get("claims")
    if not isinstance(claims, dict) or claims.get("model_valid") is not False:
        raise ValueError(f"protocol lock must retain model_valid=false: {path}")
    raw["_path"] = str(path)
    raw["_file_sha256"] = sha256(path.read_bytes()).hexdigest()
    return raw


def _load_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected object JSON: {path}")
    return raw


def _compose(
    identity: dict[str, object], labels: dict[str, object], protocol: dict[str, object]
) -> dict[str, object]:
    coverage_complete = labels["materialized_date_count"] == labels["expected_event_date_count"]
    fields = identity["field_coverage"]
    if not isinstance(fields, dict):
        raise ValueError("identity audit has invalid field_coverage")
    protocol_section = protocol.get("protocol")
    if not isinstance(protocol_section, dict):
        raise ValueError("protocol lock has no protocol section")
    reasons = [
        "StableSeat is not authorized by the identity audit",
        "industry coverage is incomplete",
        "concept coverage is incomplete",
    ]
    if not coverage_complete:
        reasons.append("V2 label coverage is incomplete across verified event dates")
    return {
        "status": "audited_fail_closed",
        "model_valid": False,
        "data_scope": "real_institutional_flow_proxy",
        "research_entity": "InstitutionFlowEvent",
        "stable_seat_authorized": False,
        "label_schema_version": labels["label_schema_version"],
        "protocol_id": protocol_section["protocol_id"],
        "source_span": {
            "trade_date_start": identity["trade_date_start"],
            "trade_date_end": identity["trade_date_end"],
            "verified_event_date_count": identity["trade_date_count"],
        },
        "v2_label_coverage": {
            "materialized_date_count": labels["materialized_date_count"],
            "expected_event_date_count": labels["expected_event_date_count"],
            "complete": coverage_complete,
            "eligible_rate_by_horizon": labels["eligible_rate_by_horizon"],
        },
        "field_coverage": fields,
        "blocked_reasons": reasons,
        "inputs": {
            "identity_audit": {"path": identity["_path"], "sha256": identity["_file_sha256"]},
            "label_audit": {"path": labels["_path"], "sha256": labels["_file_sha256"]},
            "protocol_lock": {"path": protocol["_path"], "sha256": protocol["_file_sha256"]},
        },
    }


if __name__ == "__main__":
    main()
