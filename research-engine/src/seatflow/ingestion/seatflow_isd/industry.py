"""Industry membership adapter for SeatFlow-ISD V1.

Parses the SW2021 L1 industry-membership snapshot layout produced
by the real Tushare backfill at
``<root>/r1_industry_membership/tushare/sw2021_l1_industry_membership/``:

    l1_code=801010.SI/
      is_new=N/
        data.json
        manifest.json
      is_new=Y/
        data.json
        manifest.json
    l1_code=801030.SI/
      ...

The ``data.json`` payload is a list of objects, each with at least
``l1_code``, ``l1_name``, ``ts_code`` (stock id) and ``name`` (stock
name). The adapter produces a flat list of
:class:`IndustryMembershipRow` records and a verified source
metadata object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from seatflow.dataset.isd_adapter import ISDIndustryMembership
from seatflow.domain.time import assert_aware


@dataclass(frozen=True)
class IndustryMembershipRow:
    """One (stock_id, industry_code) membership record.

    ``in_date`` / ``out_date`` are the SW2021 membership effective
    dates when present. ``is_new`` distinguishes the snapshot
    variant (N = normal, Y = newly added).
    """

    stock_id: str
    industry_code: str
    industry_name: str
    is_new: bool
    in_date: str | None
    out_date: str | None
    available_time: datetime

    def to_isd_row(self) -> ISDIndustryMembership:
        return ISDIndustryMembership(
            stock_id=self.stock_id,
            industry_code=self.industry_code,
            industry_name=self.industry_name,
            available_time=self.available_time,
            source_lineage="tushare",
            source_version="sw2021_l1_industry_membership",
        )


@dataclass(frozen=True)
class IndustrySource:
    """Verified metadata for the industry-membership source."""

    data_root: Path
    manifest_fingerprint: str
    industry_count: int
    membership_count: int
    verified: bool


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert_aware(parsed, "datetime")
    return parsed


def _l1_dir_to_industry_code(name: str) -> str:
    """Map ``l1_code=801010.SI`` to ``801010.SI``."""
    if "=" in name:
        return name.split("=", 1)[1]
    return name


def _is_new_to_bool(name: str) -> bool:
    return "=Y" in name


def _walk_l1_snapshots(
    root: Path,
) -> list[tuple[Path, Path]]:
    """Return every (data_path, manifest_path) pair under the SW2021 tree."""
    pairs: list[tuple[Path, Path]] = []
    if not root.exists():
        return pairs
    for l1_dir in sorted(root.iterdir()):
        if not l1_dir.is_dir() or not l1_dir.name.startswith("l1_code="):
            continue
        for variant in sorted(l1_dir.iterdir()):
            if not variant.is_dir() or not variant.name.startswith("is_new="):
                continue
            data = variant / "data.json"
            manifest = variant / "manifest.json"
            if data.is_file() and manifest.is_file():
                pairs.append((data, manifest))
    return pairs


def _verify_manifest(data_path: Path, manifest_path: Path) -> str:
    """Recompute the SHA-256 of data.json and compare to the manifest."""
    raw = data_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256")
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(
            f"industry manifest hash mismatch at {manifest_path}: "
            f"expected={expected!r} actual={actual!r}"
        )
    return actual


def load_industry_membership(
    source_root: Path,
) -> tuple[list[IndustryMembershipRow], IndustrySource]:
    """Load every verified SW2021 L1 industry-membership record.

    The source root must point at the parent directory that
    contains the ``sw2021_l1_industry_membership/`` subtree.
    """
    root = source_root / "sw2021_l1_industry_membership"
    if not root.exists():
        raise FileNotFoundError(
            f"no industry-membership root at {root}"
        )
    pairs = _walk_l1_snapshots(root)
    if not pairs:
        raise FileNotFoundError(f"no l1_code directories under {root}")
    rows: list[IndustryMembershipRow] = []
    fingerprint_payload: list[tuple[str, str]] = []
    for data_path, manifest_path in pairs:
        sha = _verify_manifest(data_path, manifest_path)
        fingerprint_payload.append(
            (str(data_path.relative_to(source_root)), sha)
        )
        l1_code = _l1_dir_to_industry_code(
            data_path.parent.parent.name
        )
        is_new = _is_new_to_bool(data_path.parent.name)
        for raw_row in json.loads(data_path.read_bytes()):
            in_date = raw_row.get("in_date")
            if in_date:
                available_time_str = f"{in_date}T00:00:00+00:00"
            else:
                available_time_str = "2023-01-01T00:00:00+00:00"
            try:
                available_time = _parse_dt(available_time_str)
            except (TypeError, ValueError):
                available_time = _parse_dt("2023-01-01T00:00:00+00:00")
            rows.append(
                IndustryMembershipRow(
                    stock_id=str(raw_row.get("ts_code", "")),
                    industry_code=l1_code,
                    industry_name=str(raw_row.get("l1_name", "")),
                    is_new=is_new,
                    in_date=raw_row.get("in_date"),
                    out_date=raw_row.get("out_date"),
                    available_time=available_time,
                )
            )
    rows.sort(key=lambda r: (r.industry_code, r.stock_id))
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return rows, IndustrySource(
        data_root=root,
        manifest_fingerprint=fingerprint,
        industry_count=len(pairs),
        membership_count=len(rows),
        verified=True,
    )


__all__ = [
    "IndustryMembershipRow",
    "IndustrySource",
    "load_industry_membership",
]
