"""Verify V3 package boundaries.

V3 code must never import from legacy ``src.seatflow`` or ``backend-java``.
This test scans every Python source file under ``research-engine/src`` and
asserts the forbidden patterns are absent.
"""

from pathlib import Path


def test_v3_package_does_not_import_legacy_src() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "research-engine" / "src"
    if not source_dir.exists():
        # If the directory does not exist yet, the test should FAIL.
        raise AssertionError(f"Source directory missing: {source_dir}")

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_dir.rglob("*.py")
    )
    assert "src.seatflow" not in source
    assert "backend-java" not in source


def test_v3_version_is_set() -> None:
    import seatflow

    assert seatflow.__version__ == "3.0.0-dev"
