from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/audit_seat_identity_stability.py")
    spec = importlib.util.spec_from_file_location("seat_identity_stability_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load identity audit script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_metrics_expose_repetition_without_authorizing_stable_seats() -> None:
    module = _module()
    metrics = module._identity_metrics(
        {"a": {"2023-01-03", "2023-01-04"}, "b": {"2023-01-04"}},
        ["2023-01-03", "2023-01-04"],
    )

    assert metrics["unique_identity_count"] == 2
    assert metrics["repeat_identity_ratio"] == 0.5
    assert metrics["rolling_repeat_ratio"] == {"5": 0.0, "20": 0.0, "60": 0.0}
