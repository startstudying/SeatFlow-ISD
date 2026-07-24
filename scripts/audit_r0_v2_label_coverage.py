"""Create a fail-closed audit summary from materialized V2 label artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

LABEL_SCHEMA = "executable_excess_return_v2"


def main() -> None:
    args = _arguments()
    event_dates = _event_dates(args.event_root)
    artifacts = sorted(Path(args.artifact_root).glob("R0_V2_UNIVERSE_*/result.json"))
    rows = [_load_verified(path) for path in artifacts]
    payload = _summarize(rows, event_dates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"dates": payload["materialized_date_count"], "output": str(output)}))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="artifacts/diagnostics")
    parser.add_argument(
        "--event-root",
        default=(
            "data/historical_backfill/tushare_2023_2026_v2/processed/tushare/"
            "dragon_tiger_seat_event"
        ),
    )
    parser.add_argument(
        "--output", default="artifacts/diagnostics/R0_V2_LABEL_COVERAGE_AUDIT/result.json"
    )
    return parser.parse_args()


def _event_dates(root: str) -> tuple[str, ...]:
    return tuple(sorted(path.name.removeprefix("date=") for path in Path(root).glob("date=*")))


def _load_verified(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"label artifact must be an object: {path}")
    if raw.get("status") != "evaluated_fail_closed" or raw.get("model_valid") is not False:
        raise ValueError(f"label artifact is not fail-closed: {path}")
    if raw.get("label_schema_version") != LABEL_SCHEMA:
        raise ValueError(f"label artifact schema mismatch: {path}")
    labels = raw.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"label artifact has no labels list: {path}")
    raw["_artifact_path"] = str(path)
    raw["_file_sha256"] = sha256(path.read_bytes()).hexdigest()
    return raw


def _summarize(rows: list[dict[str, object]], event_dates: tuple[str, ...]) -> dict[str, object]:
    seen_dates: set[str] = set()
    candidates = 0
    labels_by_horizon: Counter[str] = Counter()
    eligible_by_horizon: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    daily: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []

    for row in rows:
        as_of = _require_string(row, "as_of_date")
        if as_of in seen_dates:
            raise ValueError(f"duplicate V2 label artifact date: {as_of}")
        seen_dates.add(as_of)
        candidates += _require_int(row, "candidate_sample_count")
        labels = row["labels"]
        assert isinstance(labels, list)
        daily_eligible = 0
        for label in labels:
            if not isinstance(label, dict):
                raise ValueError(f"invalid label entry in {as_of}")
            horizon = str(label.get("horizon"))
            labels_by_horizon[horizon] += 1
            disposition = str(label.get("disposition"))
            dispositions[disposition] += 1
            if disposition == "eligible":
                eligible_by_horizon[horizon] += 1
                daily_eligible += 1
                directions[str(label.get("direction_label"))] += 1
        daily.append(
            {
                "as_of_date": as_of,
                "candidate_sample_count": _require_int(row, "candidate_sample_count"),
                "training_eligible_sample_count": _require_int(
                    row, "training_eligible_sample_count"
                ),
                "label_count": len(labels),
                "eligible_label_count": daily_eligible,
            }
        )
        sources.append(
            {
                "path": _require_string(row, "_artifact_path"),
                "file_sha256": _require_string(row, "_file_sha256"),
                "artifact_sha256": _require_string(row, "sha256"),
            }
        )

    missing_dates = sorted(set(event_dates) - seen_dates)
    return {
        "status": "audited_fail_closed",
        "model_valid": False,
        "label_schema_version": LABEL_SCHEMA,
        "expected_event_date_count": len(event_dates),
        "materialized_date_count": len(rows),
        "missing_event_date_count": len(missing_dates),
        "missing_event_dates": missing_dates,
        "candidate_sample_count": candidates,
        "daily": daily,
        "label_counts_by_horizon": dict(sorted(labels_by_horizon.items())),
        "eligible_label_counts_by_horizon": dict(sorted(eligible_by_horizon.items())),
        "eligible_rate_by_horizon": {
            horizon: eligible_by_horizon[horizon] / labels_by_horizon[horizon]
            for horizon in sorted(labels_by_horizon)
        },
        "direction_counts": dict(sorted(directions.items())),
        "disposition_counts": dict(sorted(dispositions.items())),
        "source_artifacts": sources,
    }


def _require_string(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact field '{field}' must be a non-empty string")
    return value


def _require_int(row: dict[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"artifact field '{field}' must be an integer")
    return value


if __name__ == "__main__":
    main()
