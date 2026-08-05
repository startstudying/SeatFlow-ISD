"""Contract tests for the SeatFlow V3 OpenAPI documents.

These tests enforce the rules documented in
``docs/api/v3-contract-rules.md``. They run without a live server: they parse
the OpenAPI documents, walk every operation, and verify that each contract
invariant holds.

Run with::

    pip install pyyaml jsonschema openapi-spec-validator pytest
    pytest contracts/tests/test_openapi_contracts.py -v

The tests are deliberately defensive: they load the YAML by hand to support
custom structural checks beyond what ``openapi-spec-validator`` would catch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "SeatFlow").exists() is False else Path(__file__).resolve().parents[2]
# The tests live at D:\SeatFlow\contracts\tests; resolve relative to that.
CONTRACTS_ROOT = Path(__file__).resolve().parents[1]
CONTROL_API = CONTRACTS_ROOT / "openapi" / "control-api.yaml"
RESEARCH_API = CONTRACTS_ROOT / "openapi" / "research-api.yaml"
TASK_EVENTS_SCHEMA = CONTRACTS_ROOT / "events" / "task-events.schema.json"

# Long-running operation suffixes used to detect async endpoints. The
# contract rules require any 2xx response on these paths to return a taskId
# in the response schema.
LONG_RUNNING_SUFFIXES = (
    "build",
    "validate",
    "training/base",
    "training/direct",
    "training/spillover",
    "training/full",
    "evaluation/walk-forward",
    "evaluation/blind",
    "backtests",
    "inference/daily",
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_operations(doc: Dict[str, Any]) -> Iterator[Tuple[str, str, str, Dict[str, Any]]]:
    """Yield (path, method, operationId, operation) for every operation in the doc."""
    for path, methods in doc.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            yield path, method.lower(), operation_id, operation


def _is_long_running(path: str) -> bool:
    """Determine whether a path is a long-running operation.

    A path is long-running if the final segment (after the version prefix)
    matches one of the documented long-running operation suffixes.
    """
    # Strip the /api/v1 or /research/v1 prefix, then take the last path segment.
    parts = [p for p in path.split("/") if p]
    # Drop non-resource prefix segments such as api, v1, research.
    filtered: List[str] = []
    for part in parts:
        if part in {"api", "v1", "research", "v3", "system", "tasks"}:
            continue
        filtered.append(part)
    if not filtered:
        return False
    tail = "/".join(filtered)
    if tail in {"build", "validate", "backtests", "inference/daily"}:
        return True
    return tail in {
        "training/base",
        "training/direct",
        "training/spillover",
        "training/full",
        "evaluation/walk-forward",
        "evaluation/blind",
    }


def _response_statuses(operation: Dict[str, Any]) -> List[str]:
    return [code for code in operation.get("responses", {}).keys()]


def _references_seatflow_error(operation: Dict[str, Any]) -> bool:
    """Check whether the operation references SeatFlowError somewhere.

    Accepts inline SeatFlowError schemas, ``$ref`` to ``SeatFlowError`` or
    ``SeatFlowErrorResponse``, and responses that point to the
    ``SeatFlowErrorResponse`` component.
    """
    haystack = json.dumps(operation, default=str)
    return (
        "SeatFlowError" in haystack
        or "SeatFlowErrorResponse" in haystack
    )


def _operation_returns_task_id(operation: Dict[str, Any]) -> bool:
    """Check whether any 2xx response schema exposes a taskId field."""
    for code, response in operation.get("responses", {}).items():
        if not str(code).startswith("2"):
            continue
        content = response.get("content", {})
        for media_type, media in content.items():
            schema = media.get("schema", {}) if isinstance(media, dict) else {}
            serialized = json.dumps(schema, default=str)
            if "taskId" in serialized:
                return True
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def control_api() -> Dict[str, Any]:
    return _load_yaml(CONTROL_API)


@pytest.fixture(scope="module")
def research_api() -> Dict[str, Any]:
    return _load_yaml(RESEARCH_API)


@pytest.fixture(scope="module")
def task_events_schema() -> Dict[str, Any]:
    return _load_json(TASK_EVENTS_SCHEMA)


# ---------------------------------------------------------------------------
# Test documents parse
# ---------------------------------------------------------------------------
def test_control_api_parses_as_yaml(control_api: Dict[str, Any]) -> None:
    assert isinstance(control_api, dict), "control-api.yaml did not parse to a mapping"
    assert "openapi" in control_api
    assert "paths" in control_api
    assert "components" in control_api


def test_research_api_parses_as_yaml(research_api: Dict[str, Any]) -> None:
    assert isinstance(research_api, dict), "research-api.yaml did not parse to a mapping"
    assert "openapi" in research_api
    assert "paths" in research_api
    assert "components" in research_api


def test_task_events_schema_parses_as_json(task_events_schema: Dict[str, Any]) -> None:
    assert isinstance(task_events_schema, dict)
    assert "properties" in task_events_schema
    assert "eventType" in task_events_schema["properties"]


# ---------------------------------------------------------------------------
# OpenAPI version + servers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "doc_path, expected_server",
    [
        (CONTROL_API, "http://localhost:8080/api/v1"),
        (RESEARCH_API, "http://localhost:8000/research/v1"),
    ],
)
def test_openapi_version_and_server(doc_path: Path, expected_server: str) -> None:
    doc = _load_yaml(doc_path)
    assert doc["openapi"] == "3.1.0", f"{doc_path.name} must declare OpenAPI 3.1.0"
    assert doc["info"]["version"] == "3.0.0-dev", f"{doc_path.name} info.version must be 3.0.0-dev"
    servers = [s.get("url") for s in doc.get("servers", [])]
    assert expected_server in servers, f"{doc_path.name} must declare server {expected_server}"


# ---------------------------------------------------------------------------
# operationId is present on every operation
# ---------------------------------------------------------------------------
def test_every_control_operation_has_operation_id(control_api: Dict[str, Any]) -> None:
    missing = []
    for path, method, operation_id, _ in _iter_operations(control_api):
        if not operation_id or not isinstance(operation_id, str):
            missing.append(f"{method.upper()} {path}")
    assert not missing, f"control-api.yaml operations missing operationId: {missing}"


def test_every_research_operation_has_operation_id(research_api: Dict[str, Any]) -> None:
    missing = []
    for path, method, operation_id, _ in _iter_operations(research_api):
        if not operation_id or not isinstance(operation_id, str):
            missing.append(f"{method.upper()} {path}")
    assert not missing, f"research-api.yaml operations missing operationId: {missing}"


def test_operation_ids_are_unique(control_api: Dict[str, Any], research_api: Dict[str, Any]) -> None:
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for doc_name, doc in [("control", control_api), ("research", research_api)]:
        for path, method, operation_id, _ in _iter_operations(doc):
            if not operation_id:
                continue
            if operation_id in seen and seen[operation_id] != f"{doc_name}:{method}:{path}":
                duplicates.append(f"{operation_id} (also defined at {doc_name} {method} {path})")
            else:
                seen[operation_id] = f"{doc_name}:{method}:{path}"
    assert not duplicates, f"Duplicate operationIds: {duplicates}"


# ---------------------------------------------------------------------------
# SeatFlowError is referenced on every operation that has an error response
# ---------------------------------------------------------------------------
def test_control_operations_reference_seatflow_error(control_api: Dict[str, Any]) -> None:
    offenders = []
    for path, method, _, operation in _iter_operations(control_api):
        if not _references_seatflow_error(operation):
            offenders.append(f"{method.upper()} {path}")
    assert not offenders, (
        "control-api.yaml operations that do not reference SeatFlowError: " + ", ".join(offenders)
    )


def test_research_operations_reference_seatflow_error(research_api: Dict[str, Any]) -> None:
    offenders = []
    for path, method, _, operation in _iter_operations(research_api):
        if not _references_seatflow_error(operation):
            offenders.append(f"{method.upper()} {path}")
    assert not offenders, (
        "research-api.yaml operations that do not reference SeatFlowError: " + ", ".join(offenders)
    )


def test_seatflow_error_schema_defined(control_api: Dict[str, Any], research_api: Dict[str, Any]) -> None:
    for doc_name, doc in [("control", control_api), ("research", research_api)]:
        schemas = doc.get("components", {}).get("schemas", {})
        assert "SeatFlowError" in schemas, f"{doc_name} must define SeatFlowError schema"
        seatflow_error = schemas["SeatFlowError"]
        required = set(seatflow_error.get("required", []))
        assert {"code", "message", "traceId"} <= required, (
            f"{doc_name} SeatFlowError must require code, message, traceId"
        )


# ---------------------------------------------------------------------------
# Long-running operations return taskId
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "doc_path",
    [CONTROL_API, RESEARCH_API],
    ids=["control", "research"],
)
def test_long_running_operations_return_task_id(doc_path: Path) -> None:
    doc = _load_yaml(doc_path)
    missing = []
    for path, method, _, operation in _iter_operations(doc):
        if not _is_long_running(path):
            continue
        if not _operation_returns_task_id(operation):
            missing.append(f"{method.upper()} {path}")
    assert not missing, (
        f"{doc_path.name} long-running operations missing taskId in response schema: " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Task status enum and transition rules
# ---------------------------------------------------------------------------
EXPECTED_TASK_STATUS_VALUES = {
    "REQUESTED",
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "BLOCKED",
}

ALLOWED_TRANSITIONS: Dict[str, set] = {
    "REQUESTED": {"QUEUED", "CANCELLED", "BLOCKED"},
    "QUEUED": {"RUNNING", "CANCELLED", "BLOCKED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED", "BLOCKED"},
    "COMPLETED": set(),
    "CANCELLED": set(),
    "FAILED": set(),
    "BLOCKED": {"REQUESTED", "QUEUED", "RUNNING", "CANCELLED"},
}

TERMINAL_STATES = {"COMPLETED", "CANCELLED", "FAILED"}


@pytest.mark.parametrize(
    "doc_path",
    [CONTROL_API, RESEARCH_API],
    ids=["control", "research"],
)
def test_task_status_enum_matches_contract(doc_path: Path) -> None:
    doc = _load_yaml(doc_path)
    schemas = doc.get("components", {}).get("schemas", {})
    task_status = schemas.get("TaskStatus", {})
    enum_values = set(task_status.get("enum", []))
    assert enum_values == EXPECTED_TASK_STATUS_VALUES, (
        f"{doc_path.name} TaskStatus enum must equal {EXPECTED_TASK_STATUS_VALUES}"
    )


def test_no_transition_leaves_terminal_state() -> None:
    """The transition graph must not allow any move out of a terminal state."""
    for state, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            assert target not in TERMINAL_STATES or state in TERMINAL_STATES, (
                f"Invalid transition {state} -> {target} escapes terminal state"
            )
        if state in TERMINAL_STATES:
            assert targets == set(), f"Terminal state {state} must have no outgoing transitions"


def test_required_transitions_are_present() -> None:
    """Spot-check that the documented transitions are present in the graph."""
    required_edges = {
        ("REQUESTED", "QUEUED"),
        ("QUEUED", "RUNNING"),
        ("RUNNING", "COMPLETED"),
        ("REQUESTED", "CANCELLED"),
        ("QUEUED", "CANCELLED"),
        ("RUNNING", "CANCELLED"),
        ("RUNNING", "FAILED"),
        ("REQUESTED", "BLOCKED"),
        ("QUEUED", "BLOCKED"),
        ("RUNNING", "BLOCKED"),
    }
    for src, dst in required_edges:
        assert dst in ALLOWED_TRANSITIONS[src], f"Missing required transition {src} -> {dst}"


# ---------------------------------------------------------------------------
# Task events schema
# ---------------------------------------------------------------------------
def test_task_events_schema_required_fields(task_events_schema: Dict[str, Any]) -> None:
    required = set(task_events_schema.get("required", []))
    expected = {"eventType", "taskId", "timestamp", "payload", "traceId"}
    assert expected <= required, f"task-events.schema.json missing required fields: {expected - required}"


def test_task_events_enum_values(task_events_schema: Dict[str, Any]) -> None:
    enum = task_events_schema["properties"]["eventType"]["enum"]
    expected = {
        "TASK_CREATED",
        "TASK_QUEUED",
        "TASK_STARTED",
        "TASK_PROGRESS",
        "TASK_COMPLETED",
        "TASK_FAILED",
        "TASK_CANCELLED",
        "TASK_BLOCKED",
    }
    assert set(enum) == expected, f"eventType enum mismatch: {set(enum) ^ expected}"


def test_task_events_timestamp_is_datetime(task_events_schema: Dict[str, Any]) -> None:
    assert task_events_schema["properties"]["timestamp"]["format"] == "date-time"
