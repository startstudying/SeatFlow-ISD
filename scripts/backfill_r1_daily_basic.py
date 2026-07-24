"""Resumably backfill audited Tushare daily_basic snapshots for R1 features."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from seatflow.data.providers import TushareHttpProvider
from seatflow.data.store import SnapshotStore

FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
    "total_share,float_share,free_share,total_mv,circ_mv"
)
SHANGHAI = ZoneInfo("Asia/Shanghai")


def main() -> None:
    args = _arguments()
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dates = [date.fromisoformat(str(value)) for value in dataset_manifest["source_as_of_dates"]]
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    provider = TushareHttpProvider()
    if not provider.status().configured:
        raise ValueError("Tushare provider is not configured for R1 daily_basic backfill")
    store = SnapshotStore(args.output_root)

    completed: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates, start=1):
        existing = _verified_existing(args.output_root, trade_date)
        if existing is not None:
            completed.append(existing)
            continue
        params = {"trade_date": trade_date.strftime("%Y%m%d")}
        raw_rows = provider._call("daily_basic", params, FIELDS)
        rows, missingness = _canonical_rows(raw_rows, trade_date)
        query_sha256 = _hash_json(
            {"api_name": "daily_basic", "params": params, "fields": FIELDS}
        )
        artifact = store.write(
            "tushare",
            "daily_basic",
            trade_date,
            rows,
            {
                "api_name": "daily_basic",
                "query_sha256": query_sha256,
                "fields": FIELDS.split(","),
                "available_time": datetime(
                    trade_date.year,
                    trade_date.month,
                    trade_date.day,
                    18,
                    tzinfo=SHANGHAI,
                ).isoformat(),
                "missingness": missingness,
            },
        )
        completed.append(
            {
                "trade_date": trade_date.isoformat(),
                "row_count": artifact.row_count,
                "sha256": artifact.sha256,
                "query_sha256": query_sha256,
                "status": "fetched",
            }
        )
        if index % 25 == 0:
            print(json.dumps({"completed": index, "total": len(dates)}), flush=True)

    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "api_name": "daily_basic",
        "fields": FIELDS.split(","),
        "dataset_manifest": str(args.dataset_manifest),
        "dataset_manifest_sha256": dataset_manifest["sha256"],
        "requested_date_count": len(dates),
        "completed_date_count": len(completed),
        "dates": completed,
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "r1_daily_basic_backfill.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "completed_dates": len(completed),
                "sha256": payload["sha256"],
            }
        )
    )


def _canonical_rows(
    raw_rows: list[dict[str, Any]], trade_date: date
) -> tuple[list[dict[str, object]], dict[str, int]]:
    fields = (
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    )
    missingness = {field: 0 for field in fields}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        stock_id = str(raw["ts_code"])
        if stock_id in seen:
            raise ValueError(f"duplicate daily_basic stock: {trade_date}:{stock_id}")
        if str(raw["trade_date"]) != trade_date.strftime("%Y%m%d"):
            raise ValueError("daily_basic returned an unexpected trade date")
        seen.add(stock_id)
        row: dict[str, object] = {
            "trade_date": trade_date.isoformat(),
            "stock_id": stock_id,
            "available_time": datetime(
                trade_date.year,
                trade_date.month,
                trade_date.day,
                18,
                tzinfo=SHANGHAI,
            ).isoformat(),
            "source_name": "tushare",
            "source_version": "daily_basic",
        }
        for field in fields:
            value = raw.get(field)
            if value in (None, ""):
                row[field] = None
                missingness[field] += 1
            else:
                row[field] = float(value)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["stock_id"])), missingness


def _verified_existing(root: Path, trade_date: date) -> dict[str, object] | None:
    directory = root / "tushare" / "daily_basic" / f"date={trade_date.isoformat()}"
    manifest_path = directory / "manifest.json"
    data_path = directory / "data.json"
    if not manifest_path.exists() or not data_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _file_sha256(data_path)
    if (
        manifest.get("dataset") != "daily_basic"
        or manifest.get("trade_date") != trade_date.isoformat()
        or manifest.get("sha256") != digest
    ):
        raise ValueError(f"existing daily_basic snapshot failed verification: {trade_date}")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict) or not metadata.get("query_sha256"):
        raise ValueError(f"existing daily_basic snapshot lacks query hash: {trade_date}")
    return {
        "trade_date": trade_date.isoformat(),
        "row_count": int(manifest["row_count"]),
        "sha256": digest,
        "query_sha256": metadata["query_sha256"],
        "status": "verified_existing",
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-dates", type=int)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


if __name__ == "__main__":
    main()
