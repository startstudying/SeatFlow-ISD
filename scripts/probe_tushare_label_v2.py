"""Record fail-closed readiness evidence for executable-label V2 data sources."""

from __future__ import annotations

import json
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

from seatflow.data.providers import ProviderError, TushareHttpProvider
from seatflow.runtime_env import load_workspace_env

OUTPUT = Path("artifacts/diagnostics/R0_TUSHARE_LABEL_V2_READINESS/result.json")


def main() -> None:
    loaded_keys = load_workspace_env()
    provider = TushareHttpProvider()
    status = provider.status()
    probe: dict[str, object] | None = None
    failure: str | None = None
    if status.configured:
        try:
            evidence = provider.fetch_adjustment_factor_evidence(
                "000001.SZ", date(2024, 1, 2), date(2024, 1, 3)
            )
            probe = {
                "stock_id": evidence.stock_id,
                "record_count": len(evidence.records),
                "query_sha256": evidence.query_sha256,
                "row_hashes": [record.row_payload_sha256 for record in evidence.records],
            }
        except ProviderError as error:
            failure = f"{type(error).__name__}: {error}"
    payload = {
        "status": (
            "verified_read_only" if probe is not None else "blocked_fail_closed"
        ),
        "model_valid": False,
        "negative_result": probe is None,
        "checked_at": datetime.now().astimezone().isoformat(),
        "loaded_key_names": list(loaded_keys),
        "provider": status.to_dict(),
        "required_sources": ["daily", "index_daily", "adj_factor", "suspend_d"],
        "probe": probe,
        "reason": failure or (
            None
            if status.configured
            else "Tushare token or explicit HTTP transport approval is unavailable"
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["sha256"] = sha256(serialized).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"result": str(OUTPUT), "status": payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
