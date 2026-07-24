"""Audit Dragon-Tiger identity evidence without promoting proxy data to StableSeat."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import median


def main() -> None:
    args = _arguments()
    payload = _audit(Path(args.event_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"dates": payload["trade_date_count"], "events": payload["event_count"]}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-root",
        default=(
            "data/historical_backfill/tushare_2023_2026_v2/processed/tushare/"
            "dragon_tiger_seat_event"
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/diagnostics/M0_SEAT_IDENTITY_STABILITY_AUDIT/result.json",
    )
    return parser.parse_args()


def _audit(root: Path) -> dict[str, object]:
    paths = sorted(root.glob("date=*/data.json"))
    if not paths:
        raise ValueError(f"no event snapshots found under {root}")
    ids_by_scope: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    rows_by_scope: Counter[str] = Counter()
    source_versions: Counter[str] = Counter()
    available_time_missing = 0
    industry_present = 0
    reason_present = 0
    concept_present = 0
    event_count = 0
    source_hashes: list[dict[str, str]] = []
    event_dates: list[str] = []

    for path in paths:
        as_of = path.parent.name.removeprefix("date=")
        event_dates.append(as_of)
        content = path.read_bytes()
        source_hashes.append({"date": as_of, "sha256": sha256(content).hexdigest()})
        rows = json.loads(content)
        if not isinstance(rows, list):
            raise ValueError(f"event snapshot is not a list: {path}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"event snapshot contains non-object row: {path}")
            scope = str(row.get("identity_scope") or "missing")
            seat_id = str(row.get("seat_id") or "").strip()
            if not seat_id:
                raise ValueError(f"event snapshot has an empty seat_id: {path}")
            ids_by_scope[scope][seat_id].add(as_of)
            rows_by_scope[scope] += 1
            source_versions[str(row.get("source_version") or "missing")] += 1
            available_time_missing += int(not row.get("available_time"))
            industry_present += int(bool(row.get("industry")))
            reason_present += int(bool(row.get("event_reason")))
            concept_present += int(bool(row.get("concepts")))
            event_count += 1

    return {
        "status": "audited_fail_closed",
        "model_valid": False,
        "stable_seat_authorized": False,
        "trade_date_count": len(event_dates),
        "trade_date_start": min(event_dates),
        "trade_date_end": max(event_dates),
        "event_count": event_count,
        "identity_scope_row_counts": dict(sorted(rows_by_scope.items())),
        "source_version_row_counts": dict(sorted(source_versions.items())),
        "identity_repeat_metrics": {
            scope: _identity_metrics(per_id_dates, event_dates)
            for scope, per_id_dates in sorted(ids_by_scope.items())
        },
        "field_coverage": {
            "available_time": 1 - (available_time_missing / event_count),
            "industry": industry_present / event_count,
            "event_reason": reason_present / event_count,
            "concepts": concept_present / event_count,
        },
        "identity_semantics": {
            "institutional_flow_proxy": "InstitutionFlowEvent only; never StableSeat",
            "named_seat": (
                "Observed source label only; StableSeat remains unauthorized without a "
                "separately licensed completeness and persistence contract"
            ),
        },
        "blocked_reasons": [
            "top_inst source scope does not independently prove complete stable-seat coverage",
            (
                "observed name repetition is not evidence of beneficial-owner or "
                "stable-position continuity"
            ),
        ],
        "snapshot_hashes": source_hashes,
        "snapshot_hashes_sha256": _hash_source_hashes(source_hashes),
    }


def _identity_metrics(per_id_dates: dict[str, set[str]], dates: list[str]) -> dict[str, object]:
    active_counts = sorted(len(values) for values in per_id_dates.values())
    return {
        "unique_identity_count": len(active_counts),
        "active_trade_day_distribution": {
            "min": active_counts[0],
            "median": median(active_counts),
            "max": active_counts[-1],
        },
        "repeat_identity_ratio": sum(count >= 2 for count in active_counts) / len(active_counts),
        "rolling_repeat_ratio": {
            str(window): _mean_rolling_repeat_ratio(per_id_dates, dates, window)
            for window in (5, 20, 60)
        },
    }


def _mean_rolling_repeat_ratio(
    per_id_dates: dict[str, set[str]], dates: list[str], window: int
) -> float:
    if len(dates) < window:
        return 0.0
    values: list[float] = []
    for end in range(window - 1, len(dates)):
        current_dates = set(dates[end - window + 1 : end + 1])
        active_counts = [
            len(identity_dates & current_dates) for identity_dates in per_id_dates.values()
        ]
        active = [count for count in active_counts if count]
        values.append(sum(count >= 2 for count in active) / len(active) if active else 0.0)
    return sum(values) / len(values)


def _hash_source_hashes(source_hashes: list[dict[str, str]]) -> str:
    payload = json.dumps(source_hashes, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
