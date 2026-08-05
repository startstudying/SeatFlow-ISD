"""Verify V3 package boundaries.

V3 code must never import from legacy ``src.seatflow`` or ``backend-java``.
This test scans every Python source file under ``research-engine/src`` and
asserts the forbidden patterns are absent from actual import statements.
String literals mentioning these names (e.g. in release gate checks) are allowed.
"""

import ast
from pathlib import Path


def _extract_imports(source: str) -> list[str]:
    """Extract all import module paths from Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_v3_package_does_not_import_legacy_src() -> None:
    root = Path(__file__).resolve().parents[2]
    source_dir = root / "research-engine" / "src"
    if not source_dir.exists():
        raise AssertionError(f"Source directory missing: {source_dir}")

    forbidden_prefixes = ("src.seatflow", "backend-java", "apps.services")
    violations: list[str] = []

    for py_file in source_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        imports = _extract_imports(content)
        for imp in imports:
            for prefix in forbidden_prefixes:
                if imp.startswith(prefix):
                    violations.append(f"{py_file}: imports '{imp}'")

    assert not violations, f"Legacy imports found: {violations}"


def test_v3_version_is_set() -> None:
    import seatflow

    assert seatflow.__version__ == "3.0.0-dev"
