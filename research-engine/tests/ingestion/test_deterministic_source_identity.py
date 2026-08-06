"""Regression tests for deterministic Bronze source identities."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from seatflow.ingestion.models import compute_payload_hash


def _record_id_in_subprocess(seed: str) -> str:
    script = """
from seatflow.ingestion.providers.akshare import AkshareProvider

row = {"日期": "2026-08-05", "代码": "000001", "收盘": 10.5}
provider = AkshareProvider(transport=lambda method, **kwargs: [row])
print(next(iter(provider.fetch_daily_bars())).source_record_id)
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    source_dir = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(source_dir)
    return subprocess.check_output(
        [sys.executable, "-c", script],
        env=env,
        text=True,
    ).strip()


def test_akshare_record_id_is_stable_across_python_hash_seeds() -> None:
    """Replacing SHA-256 identity with Python hash() must break this test."""
    first = _record_id_in_subprocess("1")
    second = _record_id_in_subprocess("8675309")

    assert first == second
    assert re.fullmatch(r"akshare:daily_bars:[0-9a-f]{64}", first)


def test_source_identity_namespaces_same_payload_by_provider_and_dataset() -> None:
    """Dropping provider/dataset namespace must cause source ID collisions."""
    models = __import__("seatflow.ingestion.models", fromlist=["compute_source_record_id"])
    compute_source_record_id = getattr(models, "compute_source_record_id")
    payload = {"代码": "000001", "日期": "2026-08-05"}

    ak_id = compute_source_record_id(
        "akshare", "daily_bars", payload, stable_key_fields=("代码", "日期")
    )
    file_id = compute_source_record_id(
        "file", "daily_bars", payload, stable_key_fields=("代码", "日期")
    )
    membership_id = compute_source_record_id(
        "akshare", "memberships", payload, stable_key_fields=("代码", "日期")
    )

    assert len({ak_id, file_id, membership_id}) == 3
    assert compute_payload_hash(payload) in models.source_identity_material(
        "akshare", "daily_bars", payload, stable_key_fields=("代码", "日期")
    )


def test_payload_hash_rejects_noncanonical_runtime_objects() -> None:
    """Falling back to repr/str must not smuggle runtime-random ordering into an ID."""
    with pytest.raises(TypeError, match="Unsupported canonical JSON value"):
        compute_payload_hash({"seat_names": {"机构专用", "营业部"}})
