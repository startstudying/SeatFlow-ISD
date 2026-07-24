"""Record fail-closed materialization attempts without conflating logs with data validity."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def main() -> None:
    args = _arguments()
    rows = _attempts(Path(args.log_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "audited_fail_closed",
        "model_valid": False,
        "audit_kind": "r0_v2_batch_attempts",
        "attempt_count": len(rows),
        "failed_attempt_count": sum(row["stderr_state"] == "recorded_failure" for row in rows),
        "attempts": rows,
    }
    payload["sha256"] = _hash(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"attempts": len(rows), "output": str(output)}))


def _attempts(log_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(log_root.glob("r0_v2_*.err.log")):
        content = path.read_bytes()
        rows.append(
            {
                "stderr_log": str(path),
                "stderr_sha256": sha256(content).hexdigest(),
                "stderr_byte_count": len(content),
                "stderr_state": "recorded_failure" if content else "no_recorded_stderr",
            }
        )
    return rows


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="artifacts/logs")
    parser.add_argument(
        "--output", default="artifacts/diagnostics/R0_V2_BATCH_ATTEMPT_AUDIT/result.json"
    )
    return parser.parse_args()


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
