"""Top-level loader entry point for SeatFlow-ISD V1.0 datasets.

This is the canonical entry point that downstream SeatFlow V3 code
should use to obtain verified V3 records from a SeatFlow-ISD V1
engineering dataset. The loader:

1. Verifies every file against the manifest SHA-256.
2. Converts raw JSONL rows into frozen V3 contracts.
3. Runs the row-level validators and surfaces a single
   :class:`ISDLoadReport` so the caller can decide whether to
   proceed, log, or block on integrity issues.

The legacy "G13 smoke test" path is replaced by this loader; the
prior 86-record / 14-day-spread / SHA256-fake-target approach is
obsolete now that 870 trading days of real Tushare data are
available at ``D:/SeatFlow/datasets/seatflow_isd_v1/<version>/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seatflow.ingestion.seatflow_isd.adapter import to_load_result
from seatflow.ingestion.seatflow_isd.contracts import ISDLoadResult
from seatflow.ingestion.seatflow_isd.reader import (
    load_isd_dataset,
    verify_manifest,
)
from seatflow.ingestion.seatflow_isd.validator import validate_all


@dataclass(frozen=True)
class ISDLoadReport:
    """Bundle of a V1 load result plus its validator verdict.

    The report never raises on validation issues; the caller can
    decide whether the failures are blocking for their use case.
    """

    result: ISDLoadResult
    validation_passed: bool
    validation_issue_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.result.dataset_version,
            "dataset_root": self.result.dataset_root,
            "manifest_hash": self.result.manifest_hash,
            "manifest_hash_verified": self.result.manifest_hash_verified,
            "row_counts": self.result.row_counts,
            "universe_size": self.result.universe.size,
            "validation_passed": self.validation_passed,
            "validation_issue_count": self.validation_issue_count,
        }


def load_isd_v1(
    dataset_root: str | Path,
    dataset_version: str,
    *,
    strict: bool = False,
) -> ISDLoadReport:
    """Load and validate a SeatFlow-ISD V1 dataset version.

    Parameters
    ----------
    dataset_root:
        Directory that contains the ``<dataset_version>/`` subdir.
    dataset_version:
        The version label, e.g. ``v1.0-20260806``.
    strict:
        When ``True``, raise :class:`ValueError` if any validator
        issue is found. Default ``False`` returns the issues in the
        report so the caller can decide.
    """
    root = Path(dataset_root)
    version_dir = root / dataset_version
    if not version_dir.is_dir():
        raise FileNotFoundError(f"dataset version not found: {version_dir}")

    verification = verify_manifest(version_dir)
    if not verification.file_hashes_verified:
        raise ValueError(
            f"manifest hash mismatch in {verification.manifest_path}: "
            f"{list(verification.mismatches)}"
        )

    raw = load_isd_dataset(root, dataset_version)
    result = to_load_result(raw)
    report = validate_all(
        result.events,
        result.participations,
        result.market_bars,
        result.samples,
        result.labels,
        result.universe,
    )
    load_report = ISDLoadReport(
        result=result,
        validation_passed=report.passed,
        validation_issue_count=len(report.issues),
    )
    if strict and not report.passed:
        raise ValueError(
            f"validation failed: {len(report.issues)} issue(s); "
            f"first={report.issues[0]}"
        )
    return load_report


__all__ = ["ISDLoadReport", "load_isd_v1"]
