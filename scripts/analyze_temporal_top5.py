from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from seatflow.evaluation.temporal_holdings import analyze_temporal_holdings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic temporal Top-5/Top-10 holding diagnostics"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON/JSONL rows")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--top-k",
        nargs="*",
        type=int,
        default=[5, 10],
        help="Top-K values to evaluate",
    )
    args = parser.parse_args()
    rows = _read_rows(args.input)
    report = analyze_temporal_holdings(rows, top_k=tuple(args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            cast("dict[str, Any]", json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [cast("dict[str, Any]", row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [cast("dict[str, Any]", row) for row in payload["rows"]]
    raise ValueError("input must be a row list or an object containing a rows field")


if __name__ == "__main__":
    raise SystemExit(main())
