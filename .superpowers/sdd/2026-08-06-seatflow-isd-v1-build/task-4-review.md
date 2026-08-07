# Task 4 Review — Dataset Orchestrator, Quality Report, and Atomic Writer

Reviewer: independent agent
Date: 2026-08-06

## Scope

- `research-engine/src/seatflow/dataset/isd_builder.py` (new)
- `research-engine/tests/dataset/test_isd_builder.py` (new)
- `research-engine/src/seatflow/dataset/__init__.py` (modified)

## Findings

### Positive

1. **Atomic directory promotion**: Builds to a sibling temp dir, then renames. Failed writes leave existing datasets intact. Verified via monkeypatch test.
2. **Fail-closed formal validity**: `dataset_valid` is never hardcoded to `True`. Six formal gates (no proxy seats, no spillover, benchmark present, concepts present, company actions present, tradeability present, source coverage complete, PIT violation free, no immature label leakage) are all derived from computed evidence. When any fails, no `dataset_seal.json` is emitted.
3. **Deterministic manifest**: Every file gets a SHA-256 hash recorded; audit recomputes and verifies. Tampered-file test confirms detection.
4. **Complete directory contract**: bronze, silver, gold, samples, labels, splits, schemas, manifests, quality_reports all populated.
5. **Quality report**: Named-seat/proxy counts are separated; excess-unavailable reason is explicit; spillover count is tracked.
6. **Label maturity**: UNMATURE labels have null returns (never zero-filled); mature labels use the corrected open-to-close / close-to-close convention.

### Concerns / Residual risk

1. **build_samples per-cycle call**: The builder iterates cycles and calls `build_samples` per cycle. This is correct but may be slow for large source windows. Acceptable for V1.0.
2. **WalkForwardPlan serialization**: Uses `as_dict()` which includes datetime objects. The JSON serializer will need to handle these in the CLI layer.
3. **No formal seal emitted in any current fixture**: Because benchmark, concepts, company actions, and tradeability evidence are never provided. This is by design per the spec — the real build will have the same formal blockers unless those sources are added.

## Verification

- Builder tests: 6 passed
- Full dataset+labels suite: 241 passed
- Ruff: clean
- Mypy: clean
