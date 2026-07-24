from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/audit_r0_v2_label_coverage.py")
    spec = importlib.util.spec_from_file_location("r0_v2_label_coverage_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load audit script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_coverage_and_explicit_unavailable_dispositions() -> None:
    module = _module()
    rows = [
        {
            "as_of_date": "2023-01-03",
            "candidate_sample_count": 2,
            "training_eligible_sample_count": 1,
            "labels": [
                {"horizon": 1, "disposition": "eligible", "direction_label": 1},
                {"horizon": 3, "disposition": "suspended", "direction_label": None},
            ],
            "_artifact_path": "first.json",
            "_file_sha256": "file-first",
            "sha256": "artifact-first",
        },
        {
            "as_of_date": "2023-01-04",
            "candidate_sample_count": 1,
            "training_eligible_sample_count": 1,
            "labels": [{"horizon": 1, "disposition": "eligible", "direction_label": -1}],
            "_artifact_path": "second.json",
            "_file_sha256": "file-second",
            "sha256": "artifact-second",
        },
    ]

    summary = module._summarize(rows, ("2023-01-03", "2023-01-04", "2023-01-05"))

    assert summary["status"] == "audited_fail_closed"
    assert summary["model_valid"] is False
    assert summary["materialized_date_count"] == 2
    assert summary["missing_event_dates"] == ["2023-01-05"]
    assert summary["candidate_sample_count"] == 3
    assert summary["eligible_rate_by_horizon"] == {"1": 1.0, "3": 0.0}
    assert summary["direction_counts"] == {"-1": 1, "1": 1}
    assert summary["disposition_counts"] == {"eligible": 2, "suspended": 1}
