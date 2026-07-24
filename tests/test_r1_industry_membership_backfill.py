from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "backfill_r1_industry_membership",
    ROOT / "scripts" / "backfill_r1_industry_membership.py",
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

ROW_LIMIT = module.ROW_LIMIT
_canonical_rows = module._canonical_rows
_guard_untruncated = module._guard_untruncated


def test_industry_membership_canonicalization_normalizes_hashes_and_sorts() -> None:
    rows = _canonical_rows(
        [
            {
                "l1_code": "801010.SI",
                "l1_name": "农林牧渔",
                "l2_code": "801012.SI",
                "l2_name": " 渔业 ",
                "l3_code": "850122.SI",
                "l3_name": "水产养殖",
                "ts_code": "000002.SZ",
                "name": "万科A",
                "in_date": "20210101",
                "out_date": "",
                "is_new": "Y",
            },
            {
                "l1_code": "801010.SI",
                "l1_name": "农林牧渔",
                "l2_code": "801011.SI",
                "l2_name": "种植业",
                "l3_code": None,
                "l3_name": None,
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "in_date": "20200102",
                "out_date": None,
                "is_new": "Y",
            },
        ],
        "801010.SI",
        "农林牧渔",
        "Y",
    )

    assert [row["stock_id"] for row in rows] == ["000001.SZ", "000002.SZ"]
    assert rows[0]["in_date"] == "2020-01-02"
    assert rows[0]["out_date"] is None
    assert rows[0]["l3_code"] is None
    assert rows[1]["l2_name"] == "渔业"
    assert rows[1]["row_payload_sha256"] != rows[0]["row_payload_sha256"]
    assert all(isinstance(row["row_payload_sha256"], str) for row in rows)


def test_industry_membership_truncation_guard_fails_at_tushare_limit() -> None:
    raw_rows = [{"ts_code": f"{index:06d}.SZ"} for index in range(ROW_LIMIT)]

    with pytest.raises(ValueError, match="row limit"):
        _guard_untruncated(raw_rows, "801010.SI", "Y")
