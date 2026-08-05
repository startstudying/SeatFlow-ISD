# SeatFlow V3 API Contract Rules

This document defines the rules every SeatFlow V3 OpenAPI contract must
follow. The rules are enforced by
`contracts/tests/test_openapi_contracts.py`. A contract change that breaks
any rule must also update the rule set and the tests.

## Artifacts

| Artifact | Path | Purpose |
| --- | --- | --- |
| Control Center API | `contracts/openapi/control-api.yaml` | Control plane: projects, datasets, runs, models, audit, system |
| Research Engine API | `contracts/openapi/research-api.yaml` | Data plane: dataset build/validate, training, evaluation, backtests, inference |
| Task Events Schema | `contracts/events/task-events.schema.json` | SSE payload schema for task lifecycle events |
| Contract Tests | `contracts/tests/test_openapi_contracts.py` | Enforces every rule in this document |

## Document-level rules

- Both OpenAPI documents must use `openapi: 3.1.0`.
- Both documents must declare `info.version: 3.0.0-dev`.
- The Control Center API must declare the server
  `http://localhost:8080/api/v1`.
- The Research Engine API must declare the server
  `http://localhost:8000/research/v1`.
- Both documents must parse as valid YAML and must contain top-level
  `openapi`, `info`, `servers`, `paths`, and `components` keys.
- The SSE schema must parse as valid JSON Schema (draft 2020-12).

## Operation rules

### operationId is mandatory

Every operation under `paths` must define a stable, non-empty
`operationId`.

- `operationId` values are unique across both documents combined.
- The id uses `lowerCamelCase` and is the source of truth for generated
  client SDKs.
- Tests: `test_every_control_operation_has_operation_id`,
  `test_every_research_operation_has_operation_id`,
  `test_operation_ids_are_unique`.

### Every error references SeatFlowError

Every operation must expose at least one non-2xx response that references
the `SeatFlowError` schema, either via the `SeatFlowErrorResponse`
component or via an inline `$ref` to `SeatFlowError`.

The `SeatFlowError` schema must be defined under
`components/schemas` in both documents and must require the fields
`code`, `message`, and `traceId`.

```yaml
components:
  schemas:
    SeatFlowError:
      type: object
      required: [code, message, traceId]
      properties:
        code:        { type: string }
        message:     { type: string }
        traceId:     { type: string }
        details:     { type: object, additionalProperties: true }
        retryable:   { type: boolean, default: false }
        remediation: { type: string }
```

- Tests: `test_control_operations_reference_seatflow_error`,
  `test_research_operations_reference_seatflow_error`,
  `test_seatflow_error_schema_defined`.

### Long-running operations return taskId

Operations whose path matches one of the long-running suffixes must
return HTTP `202 Accepted` and expose a `taskId` field in the response
schema (typically via the `TaskHandle` schema).

Long-running suffixes:

- `datasets/build`
- `datasets/validate`
- `training/base`
- `training/direct`
- `training/spillover`
- `training/full`
- `evaluation/walk-forward`
- `evaluation/blind`
- `backtests`
- `inference/daily`

```yaml
components:
  schemas:
    TaskHandle:
      type: object
      required: [taskId, status]
      properties:
        taskId:    { type: string }
        status:    { $ref: '#/components/schemas/TaskStatus' }
        acceptedAt: { type: string, format: date-time }
        pollUrl:   { type: string, format: uri }
        streamUrl: { type: string, format: uri }
```

- Test: `test_long_running_operations_return_task_id`.

## Task lifecycle rules

The `TaskStatus` enum in both documents must contain exactly the
following values:

| State | Type | Meaning |
| --- | --- | --- |
| `REQUESTED` | non-terminal | Operation submitted, awaiting scheduling |
| `QUEUED` | non-terminal | Operation accepted by the queue |
| `RUNNING` | non-terminal | Operation executing |
| `COMPLETED` | terminal | Operation finished successfully |
| `CANCELLED` | terminal | Operation cancelled by user or system |
| `FAILED` | terminal | Operation finished with an error |
| `BLOCKED` | non-terminal | Operation paused waiting on an external dependency |

### Allowed transitions

```
REQUESTED -> QUEUED
QUEUED    -> RUNNING
RUNNING   -> COMPLETED

REQUESTED -> CANCELLED
QUEUED    -> CANCELLED
RUNNING   -> CANCELLED

RUNNING   -> FAILED

REQUESTED -> BLOCKED
QUEUED    -> BLOCKED
RUNNING   -> BLOCKED

BLOCKED   -> REQUESTED | QUEUED | RUNNING | CANCELLED
```

- The linear happy path is `REQUESTED -> QUEUED -> RUNNING -> COMPLETED`.
- `CANCELLED` may be reached from any non-terminal state.
- `FAILED` may only be reached from `RUNNING`.
- `BLOCKED` may be reached from any non-terminal state and may resume to
  any non-terminal state or be cancelled.
- No transition may leave a terminal state (`COMPLETED`, `CANCELLED`,
  `FAILED`). This invariant is encoded in the transition graph in the
  contract tests and is asserted by
  `test_no_transition_leaves_terminal_state`.

### SSE task events

Task lifecycle transitions are published over Server-Sent Events. Each
SSE `data:` line is a JSON object conforming to
`contracts/events/task-events.schema.json`.

The standalone JSON Schema is the source of truth for the event payload.
Both OpenAPI documents also define a `TaskEvent` component schema that
mirrors the standalone schema (with `$ref`s repointed at
`#/components/schemas/TaskStatus`) so that the `text/event-stream`
response is self-contained. When the standalone schema changes, the
`TaskEvent` component schema must be regenerated to match.

Required fields: `eventType`, `taskId`, `timestamp`, `payload`,
`traceId`.

`eventType` values:

| eventType | Corresponds to |
| --- | --- |
| `TASK_CREATED` | transition into `REQUESTED` |
| `TASK_QUEUED` | transition into `QUEUED` |
| `TASK_STARTED` | transition into `RUNNING` |
| `TASK_PROGRESS` | in-flight progress update (no state change) |
| `TASK_COMPLETED` | transition into `COMPLETED` |
| `TASK_FAILED` | transition into `FAILED` |
| `TASK_CANCELLED` | transition into `CANCELLED` |
| `TASK_BLOCKED` | transition into `BLOCKED` |

- Tests: `test_task_events_schema_required_fields`,
  `test_task_events_enum_values`,
  `test_task_events_timestamp_is_datetime`.

## Versioning

- API path prefix is `/api/v1` (control) and `/research/v1` (research).
- Contract version is pinned in `info.version` (currently `3.0.0-dev`).
- Breaking contract changes require bumping `info.version` and updating
  this document.

## Running the tests

```bash
pip install pyyaml jsonschema openapi-spec-validator pytest
pytest contracts/tests/test_openapi_contracts.py -v
```

All tests must pass before a contract change can be merged.
