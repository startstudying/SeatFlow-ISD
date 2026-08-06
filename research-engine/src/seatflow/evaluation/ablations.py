"""Ablation matrix definitions (Task 18).

Required variants:
- Base
- Amount Only
- Seat Type
- Anonymous Seat
- Real Seat Direct
- No Graph
- Shuffled Graph
- Real Graph Spillover
- Full FDG
- Full FDG without Relation State routing
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import re


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AblationVariant(str, Enum):
    """All required ablation variants."""

    BASE = "base"
    AMOUNT_ONLY = "amount_only"
    SEAT_TYPE = "seat_type"
    ANONYMOUS_SEAT = "anonymous_seat"
    SHUFFLED_SEAT = "shuffled_seat"
    REAL_SEAT_DIRECT = "real_seat_direct"
    NO_GRAPH = "no_graph"
    SHUFFLED_GRAPH = "shuffled_graph"
    REAL_GRAPH_SPILLOVER = "real_graph_spillover"
    FULL_FDG = "full_fdg"
    FULL_FDG_NO_RS = "full_fdg_no_rs"  # without Relation State routing


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for a single ablation variant."""

    variant: AblationVariant
    use_base: bool = True
    use_direct: bool = False
    use_spillover: bool = False
    use_relation_state: bool = True
    use_seat_type: bool = True
    use_seat_amount: bool = True
    use_anonymous_seat: bool = False
    use_shuffled_graph: bool = False
    use_real_graph: bool = True
    description: str = ""


DIRECT_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant.REAL_SEAT_DIRECT,
    AblationVariant.SEAT_TYPE,
    AblationVariant.ANONYMOUS_SEAT,
    AblationVariant.SHUFFLED_SEAT,
    AblationVariant.AMOUNT_ONLY,
)

DIRECT_ABLATION_MODES: dict[AblationVariant, str] = {
    AblationVariant.REAL_SEAT_DIRECT: "real_seat",
    AblationVariant.SEAT_TYPE: "seat_type",
    AblationVariant.ANONYMOUS_SEAT: "anonymous",
    AblationVariant.SHUFFLED_SEAT: "shuffled_seat",
    AblationVariant.AMOUNT_ONLY: "amount_only",
}


@dataclass(frozen=True)
class BaseOOFArtifactBinding:
    """Seed-bound Base OOF manifest lineage."""

    seed: int
    manifest_hash: str

    def __post_init__(self) -> None:
        _require_seed("seed", self.seed)
        _require_sha256("manifest_hash", self.manifest_hash)


@dataclass(frozen=True)
class DirectAblationRunSpec:
    """Frozen lineage and execution axes for one Direct ablation variant."""

    variant: AblationVariant
    dataset_id: str
    split_id: str
    calendar_id: str
    config_hash: str
    cost_hash: str
    base_oof_artifact_bindings: tuple[BaseOOFArtifactBinding, ...]
    fold_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    mode: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_oof_artifact_bindings",
            tuple(self.base_oof_artifact_bindings),
        )
        object.__setattr__(self, "fold_ids", tuple(self.fold_ids))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.variant not in DIRECT_ABLATION_MODES:
            raise ValueError(f"variant {self.variant.value!r} is not a Direct ablation variant")
        _require_sha256("dataset_id", self.dataset_id)
        _require_sha256("split_id", self.split_id)
        _require_sha256("calendar_id", self.calendar_id)
        _require_sha256("config_hash", self.config_hash)
        _require_sha256("cost_hash", self.cost_hash)
        _require_non_empty_binding_tuple(
            "base_oof_artifact_bindings",
            self.base_oof_artifact_bindings,
        )
        _require_non_empty_tuple("fold_ids", self.fold_ids)
        if len(set(self.fold_ids)) != len(self.fold_ids):
            raise ValueError("duplicate fold_ids are not allowed")
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        for seed in self.seeds:
            _require_seed("seeds", seed)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds are not allowed")
        _validate_base_oof_bindings_cover_seeds(self.base_oof_artifact_bindings, self.seeds)
        expected_mode = DIRECT_ABLATION_MODES[self.variant]
        if self.mode and self.mode != expected_mode:
            raise ValueError("mode must match the selected variant")
        object.__setattr__(self, "mode", expected_mode)

    @property
    def base_oof_artifact_hashes(self) -> tuple[str, ...]:
        """Compatibility view of seed-bound OOF manifest hashes."""
        return tuple(binding.manifest_hash for binding in self.base_oof_artifact_bindings)


@dataclass(frozen=True)
class AblationProtocolSpec:
    """Deep-immutable shared protocol for comparable Direct ablations."""

    dataset_id: str
    split_id: str
    calendar_id: str
    config_hash: str
    cost_hash: str
    base_oof_artifact_bindings: tuple[BaseOOFArtifactBinding, ...]
    fold_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    runs: tuple[DirectAblationRunSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_oof_artifact_bindings",
            tuple(self.base_oof_artifact_bindings),
        )
        object.__setattr__(self, "fold_ids", tuple(self.fold_ids))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "runs", tuple(self.runs))
        validate_direct_ablation_protocol(self)

    @property
    def base_oof_artifact_hashes(self) -> tuple[str, ...]:
        """Compatibility view of seed-bound OOF manifest hashes."""
        return tuple(binding.manifest_hash for binding in self.base_oof_artifact_bindings)


# Pre-defined configurations for each variant
ABLATION_CONFIGS: dict[AblationVariant, AblationConfig] = {
    AblationVariant.BASE: AblationConfig(
        variant=AblationVariant.BASE,
        use_base=True,
        description="Market features only, no seat/event/graph info",
    ),
    AblationVariant.AMOUNT_ONLY: AblationConfig(
        variant=AblationVariant.AMOUNT_ONLY,
        use_base=True,
        use_direct=True,
        use_seat_amount=True,
        use_seat_type=False,
        description="Direct effect using only seat buy/sell amounts",
    ),
    AblationVariant.SEAT_TYPE: AblationConfig(
        variant=AblationVariant.SEAT_TYPE,
        use_base=True,
        use_direct=True,
        use_seat_type=True,
        use_seat_amount=False,
        description="Direct effect using only seat type (broker/institution/northbound)",
    ),
    AblationVariant.ANONYMOUS_SEAT: AblationConfig(
        variant=AblationVariant.ANONYMOUS_SEAT,
        use_base=True,
        use_direct=True,
        use_anonymous_seat=True,
        description="Direct effect with seat identity removed (anonymous)",
    ),
    AblationVariant.SHUFFLED_SEAT: AblationConfig(
        variant=AblationVariant.SHUFFLED_SEAT,
        use_base=True,
        use_direct=True,
        description="Direct effect with deterministic row-aligned shuffled seat identity",
    ),
    AblationVariant.REAL_SEAT_DIRECT: AblationConfig(
        variant=AblationVariant.REAL_SEAT_DIRECT,
        use_base=True,
        use_direct=True,
        description="Full Direct effect with real seat identity",
    ),
    AblationVariant.NO_GRAPH: AblationConfig(
        variant=AblationVariant.NO_GRAPH,
        use_base=True,
        use_direct=True,
        use_spillover=False,
        use_real_graph=False,
        description="No graph propagation at all",
    ),
    AblationVariant.SHUFFLED_GRAPH: AblationConfig(
        variant=AblationVariant.SHUFFLED_GRAPH,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_shuffled_graph=True,
        description="Spillover with shuffled graph edges (placebo)",
    ),
    AblationVariant.REAL_GRAPH_SPILLOVER: AblationConfig(
        variant=AblationVariant.REAL_GRAPH_SPILLOVER,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_real_graph=True,
        description="Spillover with real graph propagation",
    ),
    AblationVariant.FULL_FDG: AblationConfig(
        variant=AblationVariant.FULL_FDG,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_relation_state=True,
        description="Full SeatFlow-FDG with Relation State routing",
    ),
    AblationVariant.FULL_FDG_NO_RS: AblationConfig(
        variant=AblationVariant.FULL_FDG_NO_RS,
        use_base=True,
        use_direct=True,
        use_spillover=True,
        use_relation_state=False,
        description="Full FDG without Relation State routing (uniform message passing)",
    ),
}


def get_ablation_config(variant: AblationVariant) -> AblationConfig:
    """Get the configuration for an ablation variant."""
    return ABLATION_CONFIGS[variant]


def all_variants() -> list[AblationVariant]:
    """Return all ablation variants in order."""
    return list(AblationVariant)


def build_direct_ablation_protocol(
    runs: Iterable[DirectAblationRunSpec],
) -> AblationProtocolSpec:
    """Build and validate the shared Direct ablation protocol."""
    run_tuple = tuple(runs)
    if not run_tuple:
        raise ValueError("runs cannot be empty")
    _validate_direct_variants(run_tuple)
    reference = run_tuple[0]
    shared_fields = (
        "dataset_id",
        "split_id",
        "calendar_id",
        "config_hash",
        "cost_hash",
        "fold_ids",
        "seeds",
        "base_oof_artifact_bindings",
    )
    for run in run_tuple[1:]:
        for field_name in shared_fields:
            if getattr(run, field_name) != getattr(reference, field_name):
                raise ValueError(f"{field_name} must be identical across Direct ablations")

    protocol = AblationProtocolSpec(
        dataset_id=reference.dataset_id,
        split_id=reference.split_id,
        calendar_id=reference.calendar_id,
        config_hash=reference.config_hash,
        cost_hash=reference.cost_hash,
        base_oof_artifact_bindings=reference.base_oof_artifact_bindings,
        fold_ids=reference.fold_ids,
        seeds=reference.seeds,
        runs=run_tuple,
    )
    return validate_direct_ablation_protocol(protocol)


def validate_direct_ablation_protocol(
    protocol: AblationProtocolSpec,
) -> AblationProtocolSpec:
    """Fail closed unless variant/mode are the only per-run differences."""
    if not isinstance(protocol.runs, tuple):
        raise ValueError("runs must be a tuple")
    _require_sha256("dataset_id", protocol.dataset_id)
    _require_sha256("split_id", protocol.split_id)
    _require_sha256("calendar_id", protocol.calendar_id)
    _require_sha256("config_hash", protocol.config_hash)
    _require_sha256("cost_hash", protocol.cost_hash)
    _require_non_empty_binding_tuple(
        "base_oof_artifact_bindings",
        protocol.base_oof_artifact_bindings,
    )
    _require_non_empty_tuple("fold_ids", protocol.fold_ids)
    if len(set(protocol.fold_ids)) != len(protocol.fold_ids):
        raise ValueError("duplicate fold_ids are not allowed")
    if not protocol.seeds:
        raise ValueError("seeds cannot be empty")
    for seed in protocol.seeds:
        _require_seed("seeds", seed)
    if len(set(protocol.seeds)) != len(protocol.seeds):
        raise ValueError("duplicate seeds are not allowed")
    _validate_base_oof_bindings_cover_seeds(protocol.base_oof_artifact_bindings, protocol.seeds)
    for run in protocol.runs:
        if run.dataset_id != protocol.dataset_id:
            raise ValueError("dataset_id must be identical across Direct ablations")
        if run.split_id != protocol.split_id:
            raise ValueError("split_id must be identical across Direct ablations")
        if run.calendar_id != protocol.calendar_id:
            raise ValueError("calendar_id must be identical across Direct ablations")
        if run.config_hash != protocol.config_hash:
            raise ValueError("config_hash must be identical across Direct ablations")
        if run.cost_hash != protocol.cost_hash:
            raise ValueError("cost_hash must be identical across Direct ablations")
        if run.fold_ids != protocol.fold_ids:
            raise ValueError("fold_ids must be identical across Direct ablations")
        if run.seeds != protocol.seeds:
            raise ValueError("seeds must be identical across Direct ablations")
        if run.base_oof_artifact_bindings != protocol.base_oof_artifact_bindings:
            raise ValueError("base_oof_artifact_bindings must be identical across Direct ablations")
    _validate_direct_variants(protocol.runs)
    return protocol


def _validate_direct_variants(runs: tuple[DirectAblationRunSpec, ...]) -> None:
    variants = tuple(run.variant for run in runs)
    if len(set(variants)) != len(variants):
        raise ValueError("duplicate variant in Direct ablation protocol")
    missing = tuple(variant for variant in DIRECT_ABLATION_VARIANTS if variant not in variants)
    if missing:
        names = ", ".join(variant.value for variant in missing)
        raise ValueError(f"missing variant in Direct ablation protocol: {names}")


def _require_non_empty_string(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_empty_tuple(field_name: str, value: tuple[str, ...]) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")


def _require_non_empty_binding_tuple(
    field_name: str,
    value: tuple[BaseOOFArtifactBinding, ...],
) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not isinstance(item, BaseOOFArtifactBinding) for item in value):
        raise ValueError(f"{field_name} must contain BaseOOFArtifactBinding values")


def _require_sha256(field_name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must use canonical sha256:<64 lowercase hex> format")


def _require_seed(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must contain non-negative integer seeds")


def _validate_base_oof_bindings_cover_seeds(
    bindings: tuple[BaseOOFArtifactBinding, ...],
    seeds: tuple[int, ...],
) -> None:
    binding_seeds = tuple(binding.seed for binding in bindings)
    if len(set(binding_seeds)) != len(binding_seeds):
        raise ValueError("duplicate base_oof_artifact_bindings seeds are not allowed")
    binding_hashes = tuple(binding.manifest_hash for binding in bindings)
    if len(set(binding_hashes)) != len(binding_hashes):
        raise ValueError("duplicate base_oof_artifact_bindings manifest_hash values are not allowed")
    if set(binding_seeds) != set(seeds):
        raise ValueError("base_oof_artifact_bindings must cover seeds exactly")
