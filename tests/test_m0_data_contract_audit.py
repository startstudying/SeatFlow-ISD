from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/build_m0_data_contract_audit.py")
    spec = importlib.util.spec_from_file_location("m0_data_contract_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load M0 contract audit script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_audit_remains_closed_for_incomplete_labels_and_graph_fields() -> None:
    module = _module()
    identity = {
        "trade_date_start": "2023-01-03",
        "trade_date_end": "2026-07-17",
        "trade_date_count": 856,
        "field_coverage": {"industry": 0.0, "concepts": 0.0},
        "_path": "identity.json",
        "_file_sha256": "identity-hash",
    }
    labels = {
        "materialized_date_count": 10,
        "expected_event_date_count": 856,
        "label_schema_version": "executable_excess_return_v2",
        "eligible_rate_by_horizon": {"1": 0.7},
        "_path": "labels.json",
        "_file_sha256": "labels-hash",
    }
    protocol = {
        "protocol": {"protocol_id": "production-v1"},
        "_path": "protocol.json",
        "_file_sha256": "protocol-hash",
    }

    payload = module._compose(identity, labels, protocol)

    assert payload["status"] == "audited_fail_closed"
    assert payload["model_valid"] is False
    assert payload["research_entity"] == "InstitutionFlowEvent"
    assert payload["v2_label_coverage"]["complete"] is False
    assert "industry coverage is incomplete" in payload["blocked_reasons"]
