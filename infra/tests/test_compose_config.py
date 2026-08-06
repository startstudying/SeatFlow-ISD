"""Tests for the SeatFlow V3 local infrastructure configuration.

These tests validate the `docker-compose.yml` file used for local
development. They are intentionally structural: they parse the compose
file, check that all required services are present, that health checks
are defined for each service, that named volumes are wired up, that
no real credentials are hard-coded, and that the MinIO init container
creates the required buckets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures / path resolution
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = REPO_ROOT / "infra" / "compose"
COMPOSE_PATH = COMPOSE_DIR / "docker-compose.yml"
ENV_EXAMPLE_PATH = COMPOSE_DIR / ".env.example"


@pytest.fixture(scope="module")
def compose_config() -> dict[str, Any]:
    """Load and parse the docker-compose.yml file once per module."""
    assert COMPOSE_PATH.exists(), f"docker-compose.yml not found at {COMPOSE_PATH}"
    with COMPOSE_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Services that must exist and expose a health check.
REQUIRED_SERVICES: list[str] = [
    "postgres",
    "redis",
    "minio",
    "mlflow",
    "prometheus",
    "grafana",
]

# Buckets that the minio-init container must create.
REQUIRED_BUCKETS: list[str] = [
    "datasets",
    "models",
    "reports",
    "predictions",
    "backtests",
]


# ---------------------------------------------------------------------------
# 1. Required services
# ---------------------------------------------------------------------------

class TestRequiredServices:
    """All required services are declared in docker-compose.yml."""

    @pytest.mark.parametrize("service_name", REQUIRED_SERVICES)
    def test_service_present(
        self,
        compose_config: dict[str, Any],
        service_name: str,
    ) -> None:
        services = compose_config.get("services", {})
        assert service_name in services, (
            f"Required service '{service_name}' is missing from "
            f"docker-compose.yml. Found services: {sorted(services.keys())}"
        )

    def test_minio_init_present(self, compose_config: dict[str, Any]) -> None:
        """The bucket-init container is also required."""
        services = compose_config.get("services", {})
        assert "minio-init" in services, (
            "minio-init service is missing; it is required to create the "
            "datasets, models, reports, predictions and backtests buckets."
        )


# ---------------------------------------------------------------------------
# 2. Health checks
# ---------------------------------------------------------------------------

class TestHealthChecks:
    """Every service exposes a healthcheck block."""

    @pytest.mark.parametrize("service_name", REQUIRED_SERVICES)
    def test_service_has_healthcheck(
        self,
        compose_config: dict[str, Any],
        service_name: str,
    ) -> None:
        services = compose_config["services"]
        service = services[service_name]
        assert "healthcheck" in service, (
            f"Service '{service_name}' is missing a healthcheck block."
        )
        healthcheck = service["healthcheck"]
        # Compose healthchecks need a test, interval, and timeout.
        assert "test" in healthcheck, (
            f"Service '{service_name}' healthcheck must define a 'test'."
        )
        assert "interval" in healthcheck, (
            f"Service '{service_name}' healthcheck must define an 'interval'."
        )
        assert "timeout" in healthcheck, (
            f"Service '{service_name}' healthcheck must define a 'timeout'."
        )


# ---------------------------------------------------------------------------
# 3. Named volumes
# ---------------------------------------------------------------------------

class TestNamedVolumes:
    """All required named volumes are declared and used by services."""

    # State that requires persistent storage.
    STATEFUL_SERVICES: list[tuple[str, str]] = [
        ("postgres", "postgres_data"),
        ("redis", "redis_data"),
        ("minio", "minio_data"),
        ("prometheus", "prometheus_data"),
        ("grafana", "grafana_data"),
    ]

    def test_top_level_volumes_declared(
        self,
        compose_config: dict[str, Any],
    ) -> None:
        volumes = compose_config.get("volumes", {})
        assert isinstance(volumes, dict) and volumes, (
            "Top-level 'volumes' section must declare named volumes."
        )

    @pytest.mark.parametrize("service_name,volume_name", STATEFUL_SERVICES)
    def test_service_uses_named_volume(
        self,
        compose_config: dict[str, Any],
        service_name: str,
        volume_name: str,
    ) -> None:
        service = compose_config["services"][service_name]
        service_volumes = service.get("volumes", []) or []
        # Look for the named volume reference (e.g. "postgres_data:/var/lib/...").
        found = any(
            isinstance(v, str) and v.split(":")[0] == volume_name
            for v in service_volumes
        )
        assert found, (
            f"Service '{service_name}' must mount the named volume "
            f"'{volume_name}'. Found: {service_volumes}"
        )

    @pytest.mark.parametrize("service_name,volume_name", STATEFUL_SERVICES)
    def test_named_volume_declared(
        self,
        compose_config: dict[str, Any],
        service_name: str,
        volume_name: str,
    ) -> None:
        volumes = compose_config.get("volumes", {})
        assert volume_name in volumes, (
            f"Named volume '{volume_name}' is used by '{service_name}' "
            f"but is not declared in the top-level 'volumes' section."
        )


# ---------------------------------------------------------------------------
# 4. No hard-coded real credentials
# ---------------------------------------------------------------------------

class TestNoHardcodedCredentials:
    """Sensitive values must come from environment variables, not literals.

    We allow clearly-dev placeholder values (e.g. `seatflow_dev_password`)
    inside the compose file because they are only used as defaults for the
    `${VAR:-default}` substitution syntax. We reject anything that looks
    like an attempt to embed an actual production secret.
    """

    # Heuristics for "looks like a real secret".
    # These are intentionally broad: any non-placeholder-looking literal
    # that smells like a credential will trip the assertion.
    SUSPECT_PATTERNS: list[re.Pattern[str]] = [
        # 30+ char base64-ish string (e.g. AWS access key).
        re.compile(r"AKIA[0-9A-Z]{16}"),
        # Long hex / base64 strings (>= 32 chars, no spaces) - typical API keys.
        re.compile(r"(?<![\w-])[A-Za-z0-9+/=_-]{32,}(?![\w-])"),
    ]

    SUSPECT_KEYS = {
        "password",
        "passwd",
        "secret",
        "secret_key",
        "api_key",
        "token",
        "private_key",
        "access_key",
    }

    PLACEHOLDER_HINTS = (
        "dev",
        "change",
        "example",
        "placeholder",
        "your_",
        "xxx",
    )

    def _is_placeholder(self, value: str) -> bool:
        lowered = value.lower()
        return any(hint in lowered for hint in self.PLACEHOLDER_HINTS)

    def test_compose_file_does_not_contain_real_secrets(
        self,
        compose_config: dict[str, Any],
    ) -> None:
        """Walk the parsed YAML and inspect every leaf value."""

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    _walk(value, f"{path}.{key}" if path else str(key))
            elif isinstance(node, list):
                for idx, value in enumerate(node):
                    _walk(value, f"{path}[{idx}]")
            elif isinstance(node, str):
                leaf = node.strip()
                key_name = path.rsplit(".", 1)[-1].lower()
                # We are only concerned about sensitive keys.
                if not any(s in key_name for s in self.SUSPECT_KEYS):
                    return
                # Allow placeholder-looking values.
                if self._is_placeholder(leaf):
                    return
                # Allow pure variable references: ${VAR} or ${VAR:-default}.
                if leaf.startswith("${") and leaf.endswith("}"):
                    return
                # Allow empty strings.
                if not leaf:
                    return
                # Reject anything that matches a known real-secret pattern.
                for pattern in self.SUSPECT_PATTERNS:
                    if pattern.search(leaf):
                        pytest.fail(
                            f"Suspicious credential-like value at {path!r}: "
                            f"{leaf!r}. Use a placeholder or reference an "
                            f"environment variable instead."
                        )

        _walk(compose_config, "")

    def test_env_example_uses_placeholders(self) -> None:
        """The .env.example file must not commit real credentials."""
        assert ENV_EXAMPLE_PATH.exists(), (
            f".env.example not found at {ENV_EXAMPLE_PATH}"
        )
        content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            value = value.strip().strip('"').strip("'")
            if not value:
                continue
            key_lower = key.lower()
            if not any(s in key_lower for s in self.SUSPECT_KEYS):
                continue
            assert self._is_placeholder(value), (
                f".env.example key '{key}' has a non-placeholder value "
                f"'{value}'. Dev-only placeholders are required."
            )


# ---------------------------------------------------------------------------
# 5. MinIO init container creates required buckets
# ---------------------------------------------------------------------------

class TestMinioInitBuckets:
    """The minio-init container must create every required bucket."""

    def test_minio_init_references_all_buckets(
        self,
        compose_config: dict[str, Any],
    ) -> None:
        init_service = compose_config["services"]["minio-init"]
        # The buckets may be defined via entrypoint, command, or env. We
        # concatenate every string field in the service and search for
        # `mc mb` invocations.
        searchable_chunks: list[str] = []

        def _collect_strings(node: Any) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    _collect_strings(value)
            elif isinstance(node, list):
                for value in node:
                    _collect_strings(value)
            elif isinstance(node, str):
                searchable_chunks.append(node)

        _collect_strings(init_service)

        combined = "\n".join(searchable_chunks)
        for bucket in REQUIRED_BUCKETS:
            # Accept either `mc mb .../<bucket>` or `<bucket>` as a literal
            # in a multi-line entrypoint. We require the bucket to be
            # mentioned explicitly so a typo or stray rename is caught.
            assert bucket in combined, (
                f"minio-init service does not create the required bucket "
                f"'{bucket}'. Required buckets: {REQUIRED_BUCKETS}."
            )
        # All `mc mb` invocations should match the required set so we don't
        # silently create extra buckets in production.
        mb_calls = re.findall(
            r"mc\s+mb\s+(?:-p\s+)?[\w./:-]+/(\w+)",
            combined,
        )
        for bucket in mb_calls:
            assert bucket in REQUIRED_BUCKETS, (
                f"minio-init creates an unexpected bucket '{bucket}'. "
                f"Only the following buckets are allowed: {REQUIRED_BUCKETS}."
            )
