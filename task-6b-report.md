# SeatFlow V3 Task 6B Report

## Scope

- Implemented the canonical expanding walk-forward protocol in `research-engine/src/seatflow/dataset/splits.py`.
- Replaced the duplicate evaluator splitter with re-exports in `research-engine/src/seatflow/evaluation/walk_forward.py`.
- Added package-level exports in `dataset/__init__.py` and `evaluation/__init__.py`.
- Replaced the owned dataset/evaluation tests with canonical contract coverage.

No changes were made to models, features, training, lineage, seal, API, graph, or backtest modules.

## RED / GREEN

- RED: `python -m pytest tests/dataset/test_splits.py tests/evaluation/test_walk_forward.py -q` failed during collection because `CycleDisposition`, `WalkForwardConfig`, and `create_walk_forward_plan` did not exist in the worktree implementation.
- GREEN: the same focused command now passes: `14 passed`.

## Implemented Contract

- Single `WalkForwardConfig`, `WalkForwardFold`, and `WalkForwardPlan` contract.
- Expanding multi-fold generation with monotonic train sets and moving validation/test prediction windows.
- Explicit timezone-aware trading-calendar validation and deterministic canonicalization of unsorted calendars.
- Calendar duplicate, missing cycle boundary date, naive datetime, invalid config, and fingerprint tamper fail closed.
- Trading-day purge and embargo are enforced with `>= label_horizon_trading_days`.
- Cycle-level atomic classification per fold: `train`, `validation`, `test`, `blind`, `purged`, or `outside`.
- Sealed blind cycles are fixed from the calendar tail and excluded from all development folds.
- OOF prediction cycle IDs are globally unique across fold prediction segments.
- Fingerprint binds config, calendar content hash, calendar evidence fingerprint, canonical cycle records, fold boundaries, blind IDs, and per-fold classification.

## Deletion / Simplification

- Removed the evaluator's second walk-forward implementation.
- Removed the old one-fold calendar-day splitter behavior.
- Legacy `create_walk_forward_splits` remains only as a compatibility wrapper and now requires an explicit trading calendar instead of silently fabricating calendar-day splits.

## Verification

- `python -m pytest tests/dataset/test_splits.py tests/evaluation/test_walk_forward.py -q` -> `14 passed`.
- `python -m pytest tests/dataset tests/evaluation -q` -> all dataset/evaluation tests passed.
- `python -m pytest -q` -> full research-engine test suite passed.
- `python -m ruff check src/seatflow/dataset/splits.py src/seatflow/evaluation/walk_forward.py tests/dataset/test_splits.py tests/evaluation/test_walk_forward.py` -> passed.
- `python -m mypy --strict src/seatflow/dataset/splits.py src/seatflow/evaluation/walk_forward.py tests/dataset/test_splits.py tests/evaluation/test_walk_forward.py` -> passed.
- `python -m compileall -q src tests` -> passed.
- `git diff --check` -> no whitespace errors; Git reported only CRLF normalization warnings.
- Full `python -m ruff check .` was run for visibility and still fails only on pre-existing, out-of-scope unused imports in `tests/graph/test_schema.py`.

## Remaining OOF Runner Handoff

- Task 6C should consume `WalkForwardPlan.folds` and treat `fold.prediction_cycle_ids` as the eligible OOF segment.
- The runner must fit transforms/model per fold using only `fold.train_cycle_ids`.
- The runner should persist the plan fingerprint, fold payloads, dataset formal seal hash, feature schema fingerprint, git SHA, transform fingerprints, checkpoint hashes, and row coverage.
- The runner should call `validate_walk_forward_plan` before materializing OOF artifacts and add artifact-level checks for missing/duplicate rows and non-finite daily paths.
