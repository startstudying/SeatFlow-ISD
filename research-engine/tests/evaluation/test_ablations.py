"""Tests for shared Direct ablation protocol constraints."""

from dataclasses import FrozenInstanceError

import pytest

from seatflow.evaluation.ablations import (
    AblationProtocolSpec,
    AblationVariant,
    BaseOOFArtifactBinding,
    DirectAblationRunSpec,
    build_direct_ablation_protocol,
    validate_direct_ablation_protocol,
)

DATASET_HASH = "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
SPLIT_HASH = "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
CALENDAR_HASH = "sha256:9999999999999999999999999999999999999999999999999999999999999999"
CONFIG_HASH = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
COST_HASH = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
BASE_BINDINGS = (
    BaseOOFArtifactBinding(
        seed=11,
        manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    BaseOOFArtifactBinding(
        seed=23,
        manifest_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
)


def _run_spec(
    variant: AblationVariant,
    *,
    fold_ids: tuple[str, ...] = ("fold-2024-01", "fold-2024-02"),
    seeds: tuple[int, ...] = (11, 23),
    base_oof_artifact_bindings: tuple[BaseOOFArtifactBinding, ...] = BASE_BINDINGS,
    dataset_id: str = DATASET_HASH,
    split_id: str = SPLIT_HASH,
    calendar_id: str = CALENDAR_HASH,
) -> DirectAblationRunSpec:
    return DirectAblationRunSpec(
        variant=variant,
        dataset_id=dataset_id,
        split_id=split_id,
        calendar_id=calendar_id,
        config_hash=CONFIG_HASH,
        cost_hash=COST_HASH,
        base_oof_artifact_bindings=base_oof_artifact_bindings,
        fold_ids=fold_ids,
        seeds=seeds,
    )


def _direct_variants() -> tuple[AblationVariant, ...]:
    return (
        AblationVariant.REAL_SEAT_DIRECT,
        AblationVariant.SEAT_TYPE,
        AblationVariant.ANONYMOUS_SEAT,
        AblationVariant.SHUFFLED_SEAT,
        AblationVariant.AMOUNT_ONLY,
    )


def test_direct_ablation_protocol_shares_all_axes_except_variant_mode() -> None:
    """Catches per-variant protocol drift in folds, seeds, data, hashes, or costs."""
    variants = _direct_variants()

    protocol = build_direct_ablation_protocol(_run_spec(variant) for variant in variants)

    assert isinstance(protocol, AblationProtocolSpec)
    assert tuple(run.variant for run in protocol.runs) == variants
    assert tuple(run.mode for run in protocol.runs) == (
        "real_seat",
        "seat_type",
        "anonymous",
        "shuffled_seat",
        "amount_only",
    )
    assert {run.dataset_id for run in protocol.runs} == {DATASET_HASH}
    assert {run.split_id for run in protocol.runs} == {SPLIT_HASH}
    assert {run.calendar_id for run in protocol.runs} == {CALENDAR_HASH}
    assert {run.fold_ids for run in protocol.runs} == {("fold-2024-01", "fold-2024-02")}
    assert {run.seeds for run in protocol.runs} == {(11, 23)}
    assert {run.base_oof_artifact_bindings for run in protocol.runs} == {BASE_BINDINGS}
    assert validate_direct_ablation_protocol(protocol) is protocol


def test_direct_ablation_protocol_is_frozen_and_deep_immutable() -> None:
    """Catches mutable protocol containers that allow post-validation drift."""
    protocol = build_direct_ablation_protocol(_run_spec(variant) for variant in _direct_variants())

    with pytest.raises(FrozenInstanceError):
        protocol.dataset_id = "changed"  # type: ignore[misc]

    with pytest.raises(TypeError):
        protocol.runs[0].fold_ids[0] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("bad_runs", "match"),
    [
        (
            (
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(AblationVariant.ANONYMOUS_SEAT),
                _run_spec(AblationVariant.SHUFFLED_SEAT),
                _run_spec(AblationVariant.AMOUNT_ONLY),
            ),
            "duplicate variant",
        ),
        (
            (
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(AblationVariant.SEAT_TYPE),
                _run_spec(AblationVariant.ANONYMOUS_SEAT),
                _run_spec(AblationVariant.SHUFFLED_SEAT),
            ),
            "missing variant",
        ),
        (
            (
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(
                    AblationVariant.SEAT_TYPE,
                    fold_ids=("fold-2024-99", "fold-2024-100"),
                ),
                _run_spec(AblationVariant.ANONYMOUS_SEAT),
                _run_spec(AblationVariant.SHUFFLED_SEAT),
                _run_spec(AblationVariant.AMOUNT_ONLY),
            ),
            "fold_ids",
        ),
        (
            (
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(
                    AblationVariant.SEAT_TYPE,
                    seeds=(99,),
                    base_oof_artifact_bindings=(
                        BaseOOFArtifactBinding(
                            seed=99,
                            manifest_hash=(
                                "sha256:9999999999999999999999999999999999999999999999999999999999999999"
                            ),
                        ),
                    ),
                ),
                _run_spec(AblationVariant.ANONYMOUS_SEAT),
                _run_spec(AblationVariant.SHUFFLED_SEAT),
                _run_spec(AblationVariant.AMOUNT_ONLY),
            ),
            "seeds",
        ),
        (
            (
                _run_spec(AblationVariant.REAL_SEAT_DIRECT),
                _run_spec(
                    AblationVariant.SEAT_TYPE,
                    base_oof_artifact_bindings=(
                        BaseOOFArtifactBinding(
                            seed=11,
                            manifest_hash=(
                                "sha256:1111111111111111111111111111111111111111111111111111111111111111"
                            ),
                        ),
                        BaseOOFArtifactBinding(
                            seed=23,
                            manifest_hash=(
                                "sha256:2222222222222222222222222222222222222222222222222222222222222222"
                            ),
                        ),
                    ),
                ),
                _run_spec(AblationVariant.ANONYMOUS_SEAT),
                _run_spec(AblationVariant.SHUFFLED_SEAT),
                _run_spec(AblationVariant.AMOUNT_ONLY),
            ),
            "base_oof_artifact_bindings",
        ),
    ],
)
def test_direct_ablation_protocol_fails_closed_on_variant_or_shared_axis_drift(
    bad_runs: tuple[DirectAblationRunSpec, ...],
    match: str,
) -> None:
    """Catches duplicate/missing variant and shared protocol-axis drift."""
    with pytest.raises(ValueError, match=match):
        build_direct_ablation_protocol(bad_runs)


def test_direct_ablation_run_spec_rejects_missing_hash_fold_or_seed() -> None:
    """Catches incomplete protocol lineage that would make ablations incomparable."""
    with pytest.raises(ValueError, match="base_oof_artifact_bindings"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, base_oof_artifact_bindings=())
    with pytest.raises(ValueError, match="fold_ids"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, fold_ids=())
    with pytest.raises(ValueError, match="seeds"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, seeds=())


def test_direct_ablation_run_spec_rejects_noncanonical_or_duplicate_axes() -> None:
    """Catches lineage hashes or fold/seed vectors that cannot support paired comparison."""
    with pytest.raises(ValueError, match="config_hash"):
        DirectAblationRunSpec(
            variant=AblationVariant.REAL_SEAT_DIRECT,
            dataset_id=DATASET_HASH,
            split_id=SPLIT_HASH,
            calendar_id=CALENDAR_HASH,
            config_hash="sha256:not-hex",
            cost_hash=COST_HASH,
            base_oof_artifact_bindings=(
                BaseOOFArtifactBinding(
                    seed=1,
                    manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
            ),
            fold_ids=("fold-1",),
            seeds=(1,),
        )
    with pytest.raises(ValueError, match="duplicate fold_ids"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, fold_ids=("fold-1", "fold-1"))
    with pytest.raises(ValueError, match="duplicate seeds"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, seeds=(1, 1))
    with pytest.raises(ValueError, match="cover seeds"):
        _run_spec(
            AblationVariant.REAL_SEAT_DIRECT,
            seeds=(11, 23),
            base_oof_artifact_bindings=(
                BaseOOFArtifactBinding(
                    seed=11,
                    manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
            ),
        )
    with pytest.raises(ValueError, match="dataset_id"):
        _run_spec(AblationVariant.REAL_SEAT_DIRECT, dataset_id="seatflow-v3")


@pytest.mark.parametrize("bad_seed", [True, False, -1])
def test_direct_ablation_rejects_bool_or_negative_seeds(bad_seed: int) -> None:
    """Catches Python bool-as-int and negative seed lineage ambiguity."""
    with pytest.raises(ValueError, match="seed"):
        BaseOOFArtifactBinding(
            seed=bad_seed,
            manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    with pytest.raises(ValueError, match="seeds"):
        _run_spec(
            AblationVariant.REAL_SEAT_DIRECT,
            seeds=(bad_seed,),
            base_oof_artifact_bindings=(
                BaseOOFArtifactBinding(
                    seed=11,
                    manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
            ),
        )


def test_direct_ablation_specs_canonicalize_mutable_inputs_to_tuples() -> None:
    """Catches caller-owned list mutation after spec construction."""
    folds = ["fold-1", "fold-2"]
    seeds = [1, 2]
    bindings = [
        BaseOOFArtifactBinding(
            seed=1,
            manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        BaseOOFArtifactBinding(
            seed=2,
            manifest_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ),
    ]

    run = DirectAblationRunSpec(
        variant=AblationVariant.SHUFFLED_SEAT,
        dataset_id=DATASET_HASH,
        split_id=SPLIT_HASH,
        calendar_id=CALENDAR_HASH,
        config_hash=CONFIG_HASH,
        cost_hash=COST_HASH,
        base_oof_artifact_bindings=bindings,  # type: ignore[arg-type]
        fold_ids=folds,  # type: ignore[arg-type]
        seeds=seeds,  # type: ignore[arg-type]
    )
    folds[0] = "mutated"
    seeds[0] = 99
    bindings[0] = BaseOOFArtifactBinding(
        seed=99,
        manifest_hash="sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    )

    assert run.fold_ids == ("fold-1", "fold-2")
    assert run.seeds == (1, 2)
    assert run.base_oof_artifact_bindings == (
        BaseOOFArtifactBinding(
            seed=1,
            manifest_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
        BaseOOFArtifactBinding(
            seed=2,
            manifest_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ),
    )


def test_ablation_protocol_direct_construction_validates_shared_contract() -> None:
    """Catches bypassing builder validation by constructing the protocol directly."""
    good = _run_spec(AblationVariant.REAL_SEAT_DIRECT)
    drifted = _run_spec(
        AblationVariant.SEAT_TYPE,
        seeds=(99, 100),
        base_oof_artifact_bindings=(
            BaseOOFArtifactBinding(
                seed=99,
                manifest_hash="sha256:1111111111111111111111111111111111111111111111111111111111111111",
            ),
            BaseOOFArtifactBinding(
                seed=100,
                manifest_hash="sha256:2222222222222222222222222222222222222222222222222222222222222222",
            ),
        ),
    )

    with pytest.raises(ValueError, match="seeds"):
        AblationProtocolSpec(
            dataset_id=good.dataset_id,
            split_id=good.split_id,
            calendar_id=good.calendar_id,
            config_hash=good.config_hash,
            cost_hash=good.cost_hash,
            base_oof_artifact_bindings=list(good.base_oof_artifact_bindings),  # type: ignore[arg-type]
            fold_ids=list(good.fold_ids),  # type: ignore[arg-type]
            seeds=list(good.seeds),  # type: ignore[arg-type]
            runs=[good, drifted],  # type: ignore[arg-type]
        )
