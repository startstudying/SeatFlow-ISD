"""Executable deployment contract for the canonical SeatFlow V3 runtime.

The production changes protected here are accidental restoration of the
legacy sample stack, a container running as root, or a Compose definition
that no longer builds and wires the three V3 application services.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COMPOSE_PATH = REPO_ROOT / "infra" / "compose" / "docker-compose.yml"
DEPLOYMENT_COMPOSE_PATH = REPO_ROOT / "deployment" / "docker-compose.yml"

RUNTIME_SERVICES = ("research-engine", "control-center", "research-console")
EXPECTED_BUILDS = {
    "research-engine": ("../../research-engine", "Dockerfile"),
    "control-center": ("../../control-center", "Dockerfile"),
    "research-console": ("../../research-console", "Dockerfile"),
}
EXPECTED_PORTS = {
    "research-engine": "8000",
    "control-center": "8080",
    "research-console": "3000",
}
SENSITIVE_ENV_VARS = (
    "POSTGRES_PASSWORD",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "GRAFANA_ADMIN_PASSWORD",
    "SEATFLOW_SECURITY_JWT_SECRET",
)
FORBIDDEN_RUNTIME_TOKENS = (
    "backend-java",
    "python -m seatflow serve-api",
    "seatflow_data_provider",
    "../frontend",
    "../backend-java",
    "mysql:",
    "sample",
    "src/seatflow",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict), f"{path} must contain a YAML mapping"
    return document


def _docker_instructions(path: Path) -> list[tuple[str, str]]:
    """Parse top-level Dockerfile instructions, joining continuations."""

    logical_lines: list[str] = []
    buffer = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        buffer = f"{buffer} {stripped}".strip()
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        logical_lines.append(buffer)
        buffer = ""
    if buffer:
        logical_lines.append(buffer)

    parsed: list[tuple[str, str]] = []
    for line in logical_lines:
        instruction, _, value = line.partition(" ")
        parsed.append((instruction.upper(), value.strip()))
    return parsed


def _instruction_values(path: Path, name: str) -> list[str]:
    return [value for instruction, value in _docker_instructions(path) if instruction == name]


def _example_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def _compose_config_command(compose_path: Path, env_file: Path) -> list[str]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is required for Compose model regression tests")
    return [
        docker,
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_path),
        "config",
        "--format",
        "json",
    ]


@pytest.fixture(scope="module")
def canonical_compose() -> dict[str, Any]:
    return _load_yaml(CANONICAL_COMPOSE_PATH)


class TestCanonicalRuntimeServices:
    def test_only_v3_application_service_names_are_present(
        self, canonical_compose: dict[str, Any]
    ) -> None:
        services = canonical_compose["services"]
        for service_name in RUNTIME_SERVICES:
            assert service_name in services
        assert {"fastapi", "backend-java", "frontend", "mysql"}.isdisjoint(services)

    @pytest.mark.parametrize("service_name", RUNTIME_SERVICES)
    def test_runtime_build_context_resolves_to_owned_component(
        self,
        canonical_compose: dict[str, Any],
        service_name: str,
    ) -> None:
        service = canonical_compose["services"][service_name]
        build = service["build"]
        expected_context, expected_dockerfile = EXPECTED_BUILDS[service_name]
        assert build["context"] == expected_context
        assert build["dockerfile"] == expected_dockerfile
        context_path = (CANONICAL_COMPOSE_PATH.parent / expected_context).resolve()
        assert context_path == REPO_ROOT / service_name
        assert (context_path / expected_dockerfile).is_file()

    @pytest.mark.parametrize("service_name", RUNTIME_SERVICES)
    def test_runtime_service_has_operational_wiring(
        self,
        canonical_compose: dict[str, Any],
        service_name: str,
    ) -> None:
        service = canonical_compose["services"][service_name]
        assert "seatflow-net" in service["networks"]
        assert service["restart"] == "unless-stopped"
        assert service["healthcheck"]["test"]
        assert service["healthcheck"]["interval"]
        assert service["healthcheck"]["timeout"]
        assert any(
            str(port).endswith(f":{EXPECTED_PORTS[service_name]}") for port in service["ports"]
        )
        limits = service["deploy"]["resources"]["limits"]
        assert float(limits["cpus"]) > 0
        assert re.fullmatch(r"[1-9]\d*(?:[KMG]i?|[KMG]B)", str(limits["memory"]))

    def test_research_healthcheck_proves_process_response_not_model_readiness(
        self, canonical_compose: dict[str, Any]
    ) -> None:
        probe = json.dumps(
            canonical_compose["services"]["research-engine"]["healthcheck"]["test"]
        ).lower()
        assert "/research/v1/health" in probe
        assert "status" not in probe
        assert "researchcapability" not in probe

    def test_runtime_dependencies_follow_v3_service_boundaries(
        self, canonical_compose: dict[str, Any]
    ) -> None:
        services = canonical_compose["services"]
        assert services["research-engine"]["depends_on"]["redis"]["condition"] == "service_healthy"
        assert (
            services["control-center"]["depends_on"]["research-engine"]["condition"]
            == "service_healthy"
        )
        assert (
            services["research-console"]["depends_on"]["control-center"]["condition"]
            == "service_healthy"
        )
        assert (
            services["control-center"]["environment"]["SEATFLOW_RESEARCH_BASE_URL"]
            == "http://research-engine:8000/research/v1"
        )

    def test_canonical_configs_contain_no_legacy_runtime_contracts(self) -> None:
        paths = [
            CANONICAL_COMPOSE_PATH,
            DEPLOYMENT_COMPOSE_PATH,
            REPO_ROOT / "research-engine" / "Dockerfile",
            REPO_ROOT / "control-center" / "Dockerfile",
            REPO_ROOT / "research-console" / "Dockerfile",
        ]
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in paths if path.exists()
        ).lower()
        for token in FORBIDDEN_RUNTIME_TOKENS:
            assert token not in combined, f"legacy runtime token remains: {token}"


class TestProductionDockerfiles:
    @pytest.mark.parametrize("service_name", RUNTIME_SERVICES)
    def test_runtime_image_switches_to_a_non_root_user(self, service_name: str) -> None:
        dockerfile = REPO_ROOT / service_name / "Dockerfile"
        users = _instruction_values(dockerfile, "USER")
        assert users, f"{service_name} Dockerfile must declare USER"
        assert users[-1].lower() not in {"root", "0", "0:0"}

    def test_research_engine_runs_only_the_canonical_uvicorn_app(self) -> None:
        dockerfile = REPO_ROOT / "research-engine" / "Dockerfile"
        commands = _instruction_values(dockerfile, "CMD") + _instruction_values(
            dockerfile, "ENTRYPOINT"
        )
        assert commands == [
            '["uvicorn", "seatflow.api.app:app", "--host", "0.0.0.0", "--port", "8000"]'
        ]
        healthchecks = _instruction_values(dockerfile, "HEALTHCHECK")
        assert len(healthchecks) == 1
        assert "/research/v1/health" in healthchecks[0]
        assert "status" not in healthchecks[0].lower()

    @pytest.mark.parametrize(
        ("service_name", "manifest", "command_fragment"),
        (
            ("research-engine", "pyproject.toml", "uvicorn"),
            ("control-center", "pom.xml", "java"),
            ("research-console", "package-lock.json", "npm"),
        ),
    )
    def test_image_builds_from_existing_manifest_and_has_runtime_command(
        self,
        service_name: str,
        manifest: str,
        command_fragment: str,
    ) -> None:
        dockerfile = REPO_ROOT / service_name / "Dockerfile"
        copy_values = _instruction_values(dockerfile, "COPY")
        assert any(manifest in value for value in copy_values)
        commands = _instruction_values(dockerfile, "CMD") + _instruction_values(
            dockerfile, "ENTRYPOINT"
        )
        assert len(commands) == 1
        assert command_fragment in commands[0].lower()

    def test_obsolete_root_dockerfile_is_removed(self) -> None:
        assert not (REPO_ROOT / "Dockerfile").exists()


class TestDeploymentEntrypoint:
    def test_deployment_compose_is_a_thin_include_of_canonical_stack(self) -> None:
        deployment = _load_yaml(DEPLOYMENT_COMPOSE_PATH)
        assert deployment == {
            "name": "seatflow",
            "include": [{"path": "../infra/compose/docker-compose.yml"}],
        }

    def test_deployment_and_canonical_entrypoints_preserve_project_identity(self) -> None:
        canonical_env = REPO_ROOT / "infra" / "compose" / ".env.example"
        deployment_env = REPO_ROOT / "deployment" / ".env.example"
        canonical_result = subprocess.run(
            _compose_config_command(CANONICAL_COMPOSE_PATH, canonical_env),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        deployment_result = subprocess.run(
            _compose_config_command(DEPLOYMENT_COMPOSE_PATH, deployment_env),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        canonical = json.loads(canonical_result.stdout)
        deployment = json.loads(deployment_result.stdout)

        assert canonical["name"] == deployment["name"] == "seatflow"
        assert (
            canonical["networks"]["seatflow-net"]["name"]
            == deployment["networks"]["seatflow-net"]["name"]
            == "seatflow_seatflow-net"
        )

    @pytest.mark.parametrize("compose_path", (CANONICAL_COMPOSE_PATH, DEPLOYMENT_COMPOSE_PATH))
    @pytest.mark.parametrize("missing_variable", SENSITIVE_ENV_VARS)
    def test_missing_sensitive_value_rejects_compose_model(
        self,
        compose_path: Path,
        missing_variable: str,
        tmp_path: Path,
    ) -> None:
        values = _example_environment(REPO_ROOT / "infra" / "compose" / ".env.example")
        values.pop(missing_variable)
        incomplete_env = tmp_path / "incomplete.env"
        incomplete_env.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items()),
            encoding="utf-8",
        )
        process_env = os.environ.copy()
        for variable in SENSITIVE_ENV_VARS:
            process_env.pop(variable, None)

        result = subprocess.run(
            _compose_config_command(compose_path, incomplete_env),
            cwd=REPO_ROOT,
            env=process_env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert missing_variable.lower() in result.stderr.lower()
