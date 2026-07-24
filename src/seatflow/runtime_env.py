from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_PREFIXES = ("SEATFLOW_", "TUSHARE_")


def load_workspace_env(path: str | Path = ".env") -> tuple[str, ...]:
    """Load local provider settings without overriding an explicitly supplied environment."""
    env_path = Path(path)
    if not env_path.is_file():
        return ()

    loaded: list[str] = []
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env assignment on line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"invalid .env key on line {line_number}")
        if not key.startswith(_ALLOWED_PREFIXES):
            continue
        value = _strip_matching_quotes(value.strip())
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return tuple(loaded)


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
