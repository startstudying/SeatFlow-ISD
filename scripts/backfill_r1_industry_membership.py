"""Resumably backfill audited Tushare SW2021 industry membership snapshots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from seatflow.data.providers import TushareHttpProvider
from seatflow.runtime_env import load_workspace_env

API_NAME = "index_member_all"
DATASET = "sw2021_l1_industry_membership"
ROW_LIMIT = 2000
FIELDS = (
    "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,"
    "in_date,out_date,is_new"
)
MEMBERSHIP_STATES = ("Y", "N")
SW2021_L1_INDUSTRIES: tuple[tuple[str, str], ...] = (
    ("801010.SI", "农林牧渔"),
    ("801030.SI", "基础化工"),
    ("801040.SI", "钢铁"),
    ("801050.SI", "有色金属"),
    ("801080.SI", "电子"),
    ("801110.SI", "家用电器"),
    ("801120.SI", "食品饮料"),
    ("801130.SI", "纺织服饰"),
    ("801140.SI", "轻工制造"),
    ("801150.SI", "医药生物"),
    ("801160.SI", "公用事业"),
    ("801170.SI", "交通运输"),
    ("801180.SI", "房地产"),
    ("801200.SI", "商贸零售"),
    ("801210.SI", "社会服务"),
    ("801230.SI", "综合"),
    ("801710.SI", "建筑材料"),
    ("801720.SI", "建筑装饰"),
    ("801730.SI", "电力设备"),
    ("801740.SI", "国防军工"),
    ("801750.SI", "计算机"),
    ("801760.SI", "传媒"),
    ("801770.SI", "通信"),
    ("801780.SI", "银行"),
    ("801790.SI", "非银金融"),
    ("801880.SI", "汽车"),
    ("801890.SI", "机械设备"),
    ("801950.SI", "煤炭"),
    ("801960.SI", "石油石化"),
    ("801970.SI", "环保"),
    ("801980.SI", "美容护理"),
)


def main() -> None:
    args = _arguments()
    loaded_keys = load_workspace_env()
    provider = TushareHttpProvider()
    if not provider.status().configured:
        raise ValueError("Tushare provider is not configured for R1 industry membership backfill")

    output_root: Path = args.output_root
    requested = (
        SW2021_L1_INDUSTRIES[: args.max_l1]
        if args.max_l1 is not None
        else SW2021_L1_INDUSTRIES
    )
    completed: list[dict[str, object]] = []
    for l1_code, expected_l1_name in requested:
        for state in MEMBERSHIP_STATES:
            existing = _verified_existing(output_root, l1_code, state)
            if existing is not None:
                completed.append(existing)
                continue
            params = {"l1_code": l1_code, "is_new": state}
            raw_rows = list(provider._call(API_NAME, params, FIELDS))
            _guard_untruncated(raw_rows, l1_code, state)
            rows = _canonical_rows(raw_rows, l1_code, expected_l1_name, state)
            query_sha256 = _hash_json(
                {"api_name": API_NAME, "params": params, "fields": FIELDS}
            )
            snapshot = _write_snapshot(
                output_root,
                l1_code,
                expected_l1_name,
                state,
                rows,
                {
                    "api_name": API_NAME,
                    "query_sha256": query_sha256,
                    "fields": FIELDS.split(","),
                    "params": params,
                    "row_limit": ROW_LIMIT,
                },
            )
            completed.append({**snapshot, "status": "fetched"})
            print(
                json.dumps(
                    {
                        "l1_code": l1_code,
                        "is_new": state,
                        "row_count": snapshot["row_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "status": "materialized_fail_closed",
        "model_valid": False,
        "api_name": API_NAME,
        "dataset": DATASET,
        "fields": FIELDS.split(","),
        "membership_states": list(MEMBERSHIP_STATES),
        "requested_l1_count": len(requested),
        "requested_snapshot_count": len(requested) * len(MEMBERSHIP_STATES),
        "completed_snapshot_count": len(completed),
        "loaded_key_names": list(loaded_keys),
        "snapshots": completed,
        "canonical_snapshots_sha256": _hash_json(completed),
        "code_sha256": _file_sha256(Path(__file__)),
    }
    payload["sha256"] = _hash_json(payload)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_root / "r1_industry_membership_backfill.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    print(
        json.dumps(
            {"completed_snapshots": len(completed), "sha256": payload["sha256"]},
            ensure_ascii=False,
        )
    )


def _canonical_rows(
    raw_rows: list[Mapping[str, Any]],
    expected_l1_code: str,
    expected_l1_name: str,
    expected_state: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for raw in raw_rows:
        row = {
            "l1_code": _require_text(raw, "l1_code"),
            "l1_name": _require_text(raw, "l1_name"),
            "l2_code": _optional_text(raw.get("l2_code")),
            "l2_name": _optional_text(raw.get("l2_name")),
            "l3_code": _optional_text(raw.get("l3_code")),
            "l3_name": _optional_text(raw.get("l3_name")),
            "stock_id": _require_text(raw, "ts_code"),
            "stock_name": _require_text(raw, "name"),
            "in_date": _canonical_date(raw.get("in_date")),
            "out_date": _canonical_date(raw.get("out_date")),
            "is_new": _require_text(raw, "is_new"),
            "source_name": "tushare",
            "source_version": API_NAME,
        }
        if row["l1_code"] != expected_l1_code:
            raise ValueError("index_member_all returned an unexpected l1_code")
        if row["l1_name"] != expected_l1_name:
            raise ValueError("index_member_all returned an unexpected l1_name")
        if row["is_new"] != expected_state:
            raise ValueError("index_member_all returned an unexpected is_new state")
        canonical_key = (
            row["l1_code"],
            row["l2_code"],
            row["l3_code"],
            row["stock_id"],
            row["in_date"],
            row["out_date"],
            row["is_new"],
        )
        if canonical_key in seen:
            raise ValueError(f"duplicate canonical industry membership row: {canonical_key}")
        seen.add(canonical_key)
        row["row_payload_sha256"] = _hash_json(row)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["stock_id"]),
            str(row["l1_code"]),
            str(row["l2_code"] or ""),
            str(row["l3_code"] or ""),
            str(row["in_date"] or ""),
            str(row["out_date"] or ""),
            str(row["is_new"]),
        ),
    )


def _guard_untruncated(raw_rows: list[Mapping[str, Any]], l1_code: str, state: str) -> None:
    if len(raw_rows) >= ROW_LIMIT:
        raise ValueError(
            f"index_member_all hit the {ROW_LIMIT} row limit for {l1_code} is_new={state}"
        )


def _write_snapshot(
    root: Path,
    l1_code: str,
    l1_name: str,
    state: str,
    rows: list[dict[str, object]],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    directory = _snapshot_directory(root, l1_code, state)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "data.json"
    data_bytes = json.dumps(
        rows, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    data_sha256 = sha256(data_bytes).hexdigest()
    canonical_rows_sha256 = _hash_json(rows)
    _atomic_write(data_path, data_bytes)
    manifest = {
        "provider": "tushare",
        "dataset": DATASET,
        "api_name": API_NAME,
        "l1_code": l1_code,
        "l1_name": l1_name,
        "is_new": state,
        "row_count": len(rows),
        "sha256": data_sha256,
        "canonical_rows_sha256": canonical_rows_sha256,
        "written_at": datetime.now().astimezone().isoformat(),
        "metadata": dict(metadata),
    }
    _atomic_write(
        directory / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    return {
        "l1_code": l1_code,
        "l1_name": l1_name,
        "is_new": state,
        "row_count": len(rows),
        "data_path": str(data_path),
        "manifest_path": str(directory / "manifest.json"),
        "sha256": data_sha256,
        "canonical_rows_sha256": canonical_rows_sha256,
        "query_sha256": metadata["query_sha256"],
    }


def _verified_existing(root: Path, l1_code: str, state: str) -> dict[str, object] | None:
    directory = _snapshot_directory(root, l1_code, state)
    data_path = directory / "data.json"
    manifest_path = directory / "manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    data_sha256 = _file_sha256(data_path)
    canonical_rows_sha256 = _hash_json(rows)
    if (
        manifest.get("dataset") != DATASET
        or manifest.get("api_name") != API_NAME
        or manifest.get("l1_code") != l1_code
        or manifest.get("is_new") != state
        or manifest.get("sha256") != data_sha256
        or manifest.get("canonical_rows_sha256") != canonical_rows_sha256
    ):
        raise ValueError(
            f"existing industry membership snapshot failed verification: {l1_code}:{state}"
        )
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict) or not metadata.get("query_sha256"):
        raise ValueError(
            f"existing industry membership snapshot lacks query hash: {l1_code}:{state}"
        )
    return {
        "l1_code": l1_code,
        "l1_name": str(manifest["l1_name"]),
        "is_new": state,
        "row_count": int(manifest["row_count"]),
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "sha256": data_sha256,
        "canonical_rows_sha256": canonical_rows_sha256,
        "query_sha256": str(metadata["query_sha256"]),
        "status": "verified_existing",
    }


def _snapshot_directory(root: Path, l1_code: str, state: str) -> Path:
    return root / "tushare" / DATASET / f"l1_code={l1_code}" / f"is_new={state}"


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = _optional_text(row.get(field))
    if value is None:
        raise ValueError(f"index_member_all row is missing {field}")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_date(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid YYYYMMDD date: {text}")
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-l1", type=int)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
