from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

_SEAT_DETAIL_SCOPES = frozenset({"complete_seat", "institutional_proxy", "unspecified"})
_LABEL_STATUSES = frozenset({"observed", "unreleased", "mixed", "not_applicable"})
_BLIND_STATUSES = frozenset({"future_blind", "contaminated_oos", "not_applicable"})


@dataclass(frozen=True)
class ProtocolWindow:
    """A named time window with explicit label-access semantics."""

    name: str
    start: date
    end: date
    label_status: str
    blind_status: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("protocol window name must not be empty")
        if self.end < self.start:
            raise ValueError(f"protocol window '{self.name}' ends before it starts")
        if self.label_status not in _LABEL_STATUSES:
            raise ValueError(f"protocol window '{self.name}' has invalid label_status")
        if self.blind_status not in _BLIND_STATUSES:
            raise ValueError(f"protocol window '{self.name}' has invalid blind_status")
        if self.blind_status == "future_blind" and self.label_status != "unreleased":
            raise ValueError("future_blind window labels must be unreleased")

    @property
    def labels_may_be_read(self) -> bool:
        return self.label_status != "unreleased"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "label_status": self.label_status,
            "blind_status": self.blind_status,
            "labels_may_be_read": self.labels_may_be_read,
        }


@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    data_scope: str
    primary_metric: str
    baseline_models: tuple[str, ...]
    seeds: tuple[int, ...]
    hyperparameter_budget: int
    max_horizon_days: int
    purge_days: int
    embargo_days: int
    blind_start: date
    blind_end: date
    seat_detail_scope: str = "unspecified"
    label_status: str = "not_applicable"
    blind_status: str = "not_applicable"
    model_family: str = "unspecified"
    evaluation_windows: tuple[ProtocolWindow, ...] = ()

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must not be empty")
        if self.data_scope not in {"real", "synthetic"}:
            raise ValueError("protocol data_scope must be real or synthetic")
        if not self.primary_metric.strip() or not self.baseline_models or not self.seeds:
            raise ValueError("protocol requires a primary metric, baselines, and seeds")
        if self.hyperparameter_budget < 0 or self.max_horizon_days < 1:
            raise ValueError("protocol budgets and horizons must be non-negative/positive")
        if self.purge_days < self.max_horizon_days or self.embargo_days < self.max_horizon_days:
            raise ValueError("purge_days and embargo_days must cover max_horizon_days")
        if self.blind_end < self.blind_start:
            raise ValueError("blind_end must not be before blind_start")
        if self.seat_detail_scope not in _SEAT_DETAIL_SCOPES:
            raise ValueError("protocol seat_detail_scope is invalid")
        if self.label_status not in _LABEL_STATUSES:
            raise ValueError("protocol label_status is invalid")
        if self.blind_status not in _BLIND_STATUSES:
            raise ValueError("protocol blind_status is invalid")
        if not self.model_family.strip():
            raise ValueError("protocol model_family must not be empty")
        window_names = [window.name for window in self.evaluation_windows]
        if len(window_names) != len(set(window_names)):
            raise ValueError("protocol evaluation window names must be unique")
        ordered_windows = tuple(sorted(self.evaluation_windows, key=lambda window: window.start))
        for previous, current in zip(ordered_windows, ordered_windows[1:], strict=False):
            if previous.end >= current.start:
                raise ValueError("protocol evaluation windows must not overlap")
        for window in self.evaluation_windows:
            if window.blind_status == "future_blind" and window.labels_may_be_read:
                raise ValueError("future blind labels must not be readable")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["baseline_models"] = list(self.baseline_models)
        payload["seeds"] = list(self.seeds)
        payload["blind_start"] = self.blind_start.isoformat()
        payload["blind_end"] = self.blind_end.isoformat()
        payload["evaluation_windows"] = [window.to_dict() for window in self.evaluation_windows]
        return payload

    def window(self, name: str) -> ProtocolWindow:
        for window in self.evaluation_windows:
            if window.name == name:
                return window
        raise ValueError(f"protocol has no evaluation window named '{name}'")

    def assert_labels_readable(self, window_name: str) -> None:
        window = self.window(window_name)
        if not window.labels_may_be_read:
            raise ValueError(f"labels are sealed for protocol window '{window_name}'")


@dataclass(frozen=True)
class FrozenProtocol:
    path: Path
    sha256: str
    protocol: ExperimentProtocol

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "protocol_id": self.protocol.protocol_id,
            "data_scope": self.protocol.data_scope,
            "seat_detail_scope": self.protocol.seat_detail_scope,
            "label_status": self.protocol.label_status,
            "blind_status": self.protocol.blind_status,
            "model_family": self.protocol.model_family,
            "blind_start": self.protocol.blind_start.isoformat(),
            "blind_end": self.protocol.blind_end.isoformat(),
        }


def load_experiment_protocol(path: str | Path) -> ExperimentProtocol:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    section = raw.get("protocol")
    if not isinstance(section, dict):
        raise ValueError("experiment protocol config must contain a [protocol] table")
    try:
        windows = _windows_from_config(raw.get("windows", {}))
        return ExperimentProtocol(
            protocol_id=str(section["protocol_id"]),
            data_scope=str(section["data_scope"]),
            primary_metric=str(section["primary_metric"]),
            baseline_models=tuple(str(item) for item in section["baseline_models"]),
            seeds=tuple(int(item) for item in section["seeds"]),
            hyperparameter_budget=int(section["hyperparameter_budget"]),
            max_horizon_days=int(section["max_horizon_days"]),
            purge_days=int(section["purge_days"]),
            embargo_days=int(section["embargo_days"]),
            blind_start=date.fromisoformat(str(section["blind_start"])),
            blind_end=date.fromisoformat(str(section["blind_end"])),
            seat_detail_scope=str(section.get("seat_detail_scope", "unspecified")),
            label_status=str(section.get("label_status", "not_applicable")),
            blind_status=str(section.get("blind_status", "not_applicable")),
            model_family=str(section.get("model_family", "unspecified")),
            evaluation_windows=windows,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment protocol config is invalid") from error


def freeze_protocol(output_dir: str | Path, protocol: ExperimentProtocol) -> FrozenProtocol:
    root = Path(output_dir) / protocol.protocol_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.lock.json"
    payload = {
        "schema_version": 2,
        "protocol": protocol.to_dict(),
        "claims": {
            "protocol_frozen": True,
            "data_ready": False,
            "model_valid": False,
            "backtest_valid": False,
        },
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    digest = sha256(content).hexdigest()
    if path.exists() and path.read_bytes() != content:
        raise ValueError(
            "protocol is already frozen with different content; create a new protocol_id"
        )
    _atomic_write(path, content)
    return FrozenProtocol(path=path, sha256=digest, protocol=protocol)


def verify_frozen_protocol(path: str | Path) -> FrozenProtocol:
    lock_path = Path(path)
    content = lock_path.read_bytes()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("protocol lock is not valid JSON") from error
    if not isinstance(raw, dict) or raw.get("schema_version") not in {1, 2}:
        raise ValueError("protocol lock schema is invalid")
    section = raw.get("protocol")
    if not isinstance(section, dict):
        raise ValueError("protocol lock has no protocol object")
    protocol = _protocol_from_dict(section)
    return FrozenProtocol(path=lock_path, sha256=sha256(content).hexdigest(), protocol=protocol)


def _protocol_from_dict(raw: dict[str, object]) -> ExperimentProtocol:
    try:
        baseline_models = _sequence_value(raw["baseline_models"], "baseline_models")
        seeds = _sequence_value(raw["seeds"], "seeds")
        return ExperimentProtocol(
            protocol_id=str(raw["protocol_id"]),
            data_scope=str(raw["data_scope"]),
            primary_metric=str(raw["primary_metric"]),
            baseline_models=tuple(str(item) for item in baseline_models),
            seeds=tuple(int(str(item)) for item in seeds),
            hyperparameter_budget=int(str(raw["hyperparameter_budget"])),
            max_horizon_days=int(str(raw["max_horizon_days"])),
            purge_days=int(str(raw["purge_days"])),
            embargo_days=int(str(raw["embargo_days"])),
            blind_start=date.fromisoformat(str(raw["blind_start"])),
            blind_end=date.fromisoformat(str(raw["blind_end"])),
            seat_detail_scope=str(raw.get("seat_detail_scope", "unspecified")),
            label_status=str(raw.get("label_status", "not_applicable")),
            blind_status=str(raw.get("blind_status", "not_applicable")),
            model_family=str(raw.get("model_family", "unspecified")),
            evaluation_windows=_windows_from_lock(raw.get("evaluation_windows", [])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("protocol lock contains invalid values") from error


def _sequence_value(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"protocol {name} must be a list")
    return value


def _windows_from_config(raw: object) -> tuple[ProtocolWindow, ...]:
    if raw == {}:
        return ()
    if not isinstance(raw, dict):
        raise ValueError("experiment protocol windows must be a table")
    windows: list[ProtocolWindow] = []
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"experiment protocol window '{name}' must be a table")
        windows.append(
            ProtocolWindow(
                name=str(name),
                start=date.fromisoformat(str(value["start"])),
                end=date.fromisoformat(str(value["end"])),
                label_status=str(value["label_status"]),
                blind_status=str(value["blind_status"]),
            )
        )
    return tuple(windows)


def _windows_from_lock(raw: object) -> tuple[ProtocolWindow, ...]:
    if raw == []:
        return ()
    if not isinstance(raw, list):
        raise ValueError("protocol evaluation_windows must be a list")
    windows: list[ProtocolWindow] = []
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("protocol evaluation_windows contains an invalid window")
        windows.append(
            ProtocolWindow(
                name=str(value["name"]),
                start=date.fromisoformat(str(value["start"])),
                end=date.fromisoformat(str(value["end"])),
                label_status=str(value["label_status"]),
                blind_status=str(value["blind_status"]),
            )
        )
    return tuple(windows)


def _atomic_write(path: Path, content: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
