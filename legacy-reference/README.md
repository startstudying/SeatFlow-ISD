# Legacy Reference

This directory points to the pre-V3 SeatFlow codebase preserved under the
immutable Git tag `legacy-before-seatflow-v3`.

## Purpose

The legacy tag exists solely for historical reference. The V3 rewrite does not
import, extend, or depend on any legacy module. Researchers may consult the tag
to understand past design decisions, negative results, and data-source adapter
experience.

## Access

```bash
git show legacy-before-seatflow-v3:<path>
```

## Boundary

- V3 code must **never** import from `src.seatflow` (legacy) or `backend-java`.
- The boundary test at `research-engine/tests/test_package_boundary.py`
  enforces this constraint at CI time.
- If you need a concept from the legacy code, re-implement it cleanly under the
  V3 architecture defined in `docs/architecture/seatflow-v3-design.md`.
