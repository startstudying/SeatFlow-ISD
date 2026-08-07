# Task 5 Review — Build/Audit CLI

Reviewer: independent agent
Date: 2026-08-06

## Scope

- `research-engine/src/seatflow/dataset/isd_cli.py` (new)
- `research-engine/tests/cli/test_isd_cli.py` (new)
- `research-engine/pyproject.toml` (modified — added `[project.scripts]`)
- `research-engine/README.md` (modified — added CLI usage docs)

## Findings

### Positive

1. **Correct exit codes**: 0 for success, 2 for invalid args / corrupt inputs, 3 for `--require-formal` with formal validity blocked.
2. **Structured JSON output**: Build and audit both emit deterministic JSON to stdout with `default=str` for datetime serialization.
3. **Error messages on stderr**: No build internals leak into stdout; diagnostics go to stderr.
4. **CLI tests cover all three exit codes**: build-0, build-2 (missing source), build-3 (require-formal), audit-0, audit-2 (missing root), audit-3 (require-formal).
5. **Console script registered**: `seatflow-isd` entry point in pyproject.toml.

### Concerns

1. **WalkForwardPlan serialization**: The `as_dict()` output contains datetime objects which `json.dumps(default=str)` handles. This is correct but not ideal for long-term reproducibility — should be canonicalized to ISO strings in a future iteration.
2. **No `--dry-run` mode**: The CLI always executes the full build. Acceptable for V1.0.

## Verification

- CLI tests: 6 passed
- Ruff: clean
- Mypy: clean
