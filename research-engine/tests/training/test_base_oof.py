"""Task 6C tests for strict Base OOF runner and immutable artifacts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from seatflow.artifacts.atomic import atomic_write_json, atomic_write_jsonl, load_jsonl_artifact
from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.splits import WalkForwardConfig, create_walk_forward_plan
from seatflow.features.registry import FeatureScope, FeatureSpec, SourceKind
from seatflow.models.base import BaseFeatureBatch, BaseFeatureContract, BasePrediction
from seatflow.training.base_oof import BaseOOFRunner, BaseTargetRow, FoldTrainingBundle, OOFInputRow
from seatflow.training.base_oof_manifest import (
    FoldEvidence,
    OOFManifest,
    OOFPredictionRow,
    validate_oof_artifacts,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _spec() -> FeatureSpec:
    return FeatureSpec(
        name="stock_volatility_5d",
        version="1.0",
        horizons=[5],
        dependencies=[],
        description="fixture",
        scope=FeatureScope.STOCK_PIT,
        source=SourceKind.STOCK_PIT,
    )


def _calendar(days: int = 85) -> tuple[datetime, ...]:
    start = datetime(2026, 1, 1, 15, tzinfo=UTC)
    return tuple(start + timedelta(days=i) for i in range(days))


def _cycles(calendar: tuple[datetime, ...]) -> tuple[EventCycle, ...]:
    return tuple(
        EventCycle(
            cycle_id=f"cycle-{i:03d}",
            stock_id=f"stock-{i % 3}",
            cycle_start=day,
            last_event_date=day,
            cycle_end=day,
            event_count=1,
            status=CycleStatus.EXPIRED,
        )
        for i, day in enumerate(calendar[:-5])
    )


def _plan(calendar: tuple[datetime, ...], cycles: tuple[EventCycle, ...]):
    return create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=WalkForwardConfig(
            min_train_trading_days=10,
            validation_trading_days=5,
            test_trading_days=3,
            label_horizon_trading_days=5,
            purge_trading_days=5,
            embargo_trading_days=5,
            blind_trading_days=0,
        ),
        calendar_evidence_fingerprint=SHA_C,
    )


def _rows_targets(
    calendar: tuple[datetime, ...],
    cycles: tuple[EventCycle, ...],
) -> tuple[tuple[OOFInputRow, ...], tuple[BaseTargetRow, ...]]:
    rows: list[OOFInputRow] = []
    targets: list[BaseTargetRow] = []
    for idx, cycle in enumerate(cycles):
        cutoff = cycle.cycle_end
        assert cutoff is not None
        prediction_time = calendar[idx + 1]
        horizon_end = calendar[idx + 5]
        rows.append(
            OOFInputRow(
                sample_id=f"sample-{idx:03d}",
                cycle_id=cycle.cycle_id,
                stock_id=cycle.stock_id,
                cutoff_time=cutoff,
                prediction_time=prediction_time,
                feature_values={"stock_volatility_5d": float(idx)},
                dataset_seal_hash=SHA_A,
                materialization_hash=SHA_B,
                lineage_hash=SHA_C,
            )
        )
        targets.append(
            BaseTargetRow(
                sample_id=f"sample-{idx:03d}",
                daily_excess_return_path=(0.1, 0.2, 0.3, 0.4, 0.5),
                evaluation_time=prediction_time,
                horizon_end_time=horizon_end,
                target_artifact_hash=SHA_D,
            )
        )
    return tuple(rows), tuple(targets)


def _development_targets(plan, rows, targets) -> tuple[BaseTargetRow, ...]:
    development_cycles = set().union(
        *(
            fold.train_cycle_ids | fold.prediction_cycle_ids
            for fold in plan.folds
        )
    )
    development_sample_ids = {
        row.sample_id for row in rows if row.cycle_id in development_cycles
    }
    return tuple(target for target in targets if target.sample_id in development_sample_ids)


class RecordingTransformer:
    def __init__(self, contract: BaseFeatureContract) -> None:
        self.contract = contract
        self.fit_sample_ids: tuple[str, ...] = ()
        self.train_mean = 0.0
        self.fingerprint = "sha256:" + SHA_E

    def fit(self, rows: tuple[OOFInputRow, ...]) -> "RecordingTransformer":
        self.fit_sample_ids = tuple(row.sample_id for row in rows)
        self.train_mean = sum(row.feature_values["stock_volatility_5d"] for row in rows) / len(rows)
        self.fingerprint = "sha256:" + f"{int(self.train_mean * 1000):064x}"[-64:]
        return self

    def transform(self, rows: tuple[OOFInputRow, ...]) -> BaseFeatureBatch:
        values = [[row.feature_values["stock_volatility_5d"] - self.train_mean] for row in rows]
        return self.contract.validate_batch(
            torch.tensor(values, dtype=torch.float32),
            [row.stock_id for row in rows],
            feature_names=self.contract.feature_names,
        )


class RecordingTrainer:
    def __init__(self) -> None:
        self.fit_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fit(
        self,
        *,
        fold_id: int,
        train_batch: BaseFeatureBatch,
        train_targets: tuple[BaseTargetRow, ...],
        train_sample_ids: tuple[str, ...],
        transform_fingerprint: str,
        seed: int,
    ) -> FoldTrainingBundle:
        assert train_batch.features.shape[0] == len(train_sample_ids)
        assert len(train_targets) == len(train_sample_ids)
        self.fit_calls.append((train_sample_ids, tuple(target.sample_id for target in train_targets)))
        return FoldTrainingBundle(
            fold_id=fold_id,
            train_sample_ids=train_sample_ids,
            transform_fingerprint=transform_fingerprint,
            checkpoint_sha256=SHA_F,
            predictor=ConstantPredictor(),
            seed=seed,
        )


class ConstantPredictor:
    def predict(
        self,
        batch: BaseFeatureBatch,
        *,
        rows: tuple[OOFInputRow, ...],
    ) -> tuple[BasePrediction, ...]:
        return tuple(
            BasePrediction(
                stock_id=row.stock_id,
                prediction_time=row.prediction_time,
                cutoff_time=row.cutoff_time,
                daily_path=(0.01, 0.02, 0.03, 0.04, 0.05),
                feature_schema_fingerprint=batch.schema_fingerprint,
            )
            for row in rows
        )


def _run_fixture(tmp_path: Path):
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    targets = _development_targets(plan, rows, all_targets)
    contract = BaseFeatureContract((_spec(),))
    transformers: list[RecordingTransformer] = []
    trainer = RecordingTrainer()

    def transformer_factory() -> RecordingTransformer:
        transformer = RecordingTransformer(contract)
        transformers.append(transformer)
        return transformer

    manifest, predictions = BaseOOFRunner(
        feature_contract=contract,
        transformer_factory=transformer_factory,
        trainer=trainer,
        output_dir=tmp_path,
        seed=7,
        config_hash=SHA_E,
        git_hash=SHA_F,
        audit_attestation_hash=SHA_B,
    ).run(
        rows=rows,
        targets=targets,
        cycles=cycles,
        trading_calendar=calendar,
        plan=plan,
        synthetic=True,
    )
    return plan, rows, manifest, predictions, transformers, trainer


def test_input_target_and_manifest_dtos_are_deeply_immutable() -> None:
    row = OOFInputRow(
        sample_id="s1",
        cycle_id="c1",
        stock_id="stock-a",
        cutoff_time=datetime(2026, 1, 1, 15, tzinfo=UTC),
        prediction_time=datetime(2026, 1, 2, 15, tzinfo=UTC),
        feature_values={"stock_volatility_5d": 1.0},
        dataset_seal_hash=SHA_A,
        materialization_hash=SHA_B,
        lineage_hash=SHA_C,
    )
    assert tuple(row.feature_values) == ("stock_volatility_5d",)
    with pytest.raises(TypeError):
        row.feature_values["stock_volatility_5d"] = 2.0
    with pytest.raises(FrozenInstanceError):
        row.sample_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        BaseTargetRow(
            sample_id="s1",
            daily_excess_return_path=(0, 0, 0, 0, 0),
            evaluation_time=datetime(2026, 1, 2),
            horizon_end_time=datetime(2026, 1, 8, tzinfo=UTC),
            target_artifact_hash=SHA_D,
        )


def test_runner_is_fold_local_exactly_once_and_does_not_write_y_true(tmp_path: Path) -> None:
    plan, rows, manifest, predictions, transformers, trainer = _run_fixture(tmp_path)
    predicted_ids = [row.sample_id for row in predictions]
    expected_cycle_ids = set().union(*(fold.prediction_cycle_ids for fold in plan.folds))
    expected_ids = {row.sample_id for row in rows if row.cycle_id in expected_cycle_ids}
    assert set(predicted_ids) == expected_ids
    assert len(predicted_ids) == len(set(predicted_ids))
    assert len(transformers) == len(plan.folds)
    assert len(trainer.fit_calls) == len(plan.folds)

    row_by_cycle = {row.cycle_id: row for row in rows}
    for fold, transformer, fit_call in zip(plan.folds, transformers, trainer.fit_calls):
        train_ids = tuple(row_by_cycle[cycle_id].sample_id for cycle_id in sorted(fold.train_cycle_ids))
        assert transformer.fit_sample_ids == train_ids
        assert fit_call[0] == train_ids
        assert fit_call[1] == train_ids
        fold_pred_ids = {row.sample_id for row in rows if row.cycle_id in fold.prediction_cycle_ids}
        assert not set(train_ids) & fold_pred_ids

    prediction_path = tmp_path / "base_oof_predictions.jsonl"
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    assert lines
    assert all("y_true" not in json.loads(line) for line in lines)
    assert manifest.classification == "SYNTHETIC_NON_FORMAL"
    assert manifest.coverage["predicted_count"] == len(predictions)
    assert manifest.coverage["eligible_count"] == len(predictions)


def test_runner_rejects_global_transformer_that_returns_foreign_contract_batch(tmp_path: Path) -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    targets = _development_targets(plan, rows, all_targets)
    runner_contract = BaseFeatureContract((_spec(),))
    foreign_contract = BaseFeatureContract(
        (
            FeatureSpec(
                name="market_excess_return_1d",
                version="1.0",
                horizons=[1],
                dependencies=[],
                description="foreign",
                scope=FeatureScope.MARKET_PIT,
                source=SourceKind.MARKET_PIT,
            ),
        )
    )

    class ForeignTransformer(RecordingTransformer):
        def __init__(self) -> None:
            super().__init__(foreign_contract)

    with pytest.raises(ValueError, match="exact BaseFeatureContract"):
        BaseOOFRunner(
            feature_contract=runner_contract,
            transformer_factory=ForeignTransformer,
            trainer=RecordingTrainer(),
            output_dir=tmp_path,
            seed=7,
            config_hash=SHA_E,
            git_hash=SHA_F,
            audit_attestation_hash=SHA_B,
        ).run(
            rows=rows,
            targets=targets,
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=True,
        )


def test_runner_fails_closed_on_duplicate_rows_or_unaligned_targets(tmp_path: Path) -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    targets = _development_targets(plan, rows, all_targets)
    contract = BaseFeatureContract((_spec(),))
    runner = BaseOOFRunner(
        feature_contract=contract,
        transformer_factory=lambda: RecordingTransformer(contract),
        trainer=RecordingTrainer(),
        output_dir=tmp_path,
        seed=7,
        config_hash=SHA_E,
        git_hash=SHA_F,
        audit_attestation_hash=SHA_B,
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        runner.run(
            rows=(rows[0], rows[0], *rows[1:]),
            targets=targets,
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=True,
        )
    with pytest.raises(ValueError, match="target sample ids"):
        runner.run(
            rows=rows,
            targets=targets[:-1],
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=True,
        )


def test_runner_blocks_public_formal_mode_even_with_caller_supplied_hashes(tmp_path: Path) -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    targets = _development_targets(plan, rows, all_targets)
    contract = BaseFeatureContract((_spec(),))
    with pytest.raises(ValueError, match="FORMAL_AUDIT_ATTESTATION_REQUIRED"):
        BaseOOFRunner(
            feature_contract=contract,
            transformer_factory=lambda: RecordingTransformer(contract),
            trainer=RecordingTrainer(),
            output_dir=tmp_path,
            seed=7,
            config_hash=SHA_E,
            git_hash=SHA_F,
            audit_attestation_hash=SHA_B,
        ).run(
            rows=rows,
            targets=targets,
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=False,
        )


def test_runner_supports_multiple_samples_per_atomic_cycle(tmp_path: Path) -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    eligible_cycle = next(iter(plan.folds[0].validation_cycle_ids))
    original = next(row for row in rows if row.cycle_id == eligible_cycle)
    extra = OOFInputRow(
        sample_id=f"{original.sample_id}-active-day-2",
        cycle_id=original.cycle_id,
        stock_id=original.stock_id,
        cutoff_time=original.cutoff_time,
        prediction_time=original.prediction_time + timedelta(hours=1),
        feature_values=original.feature_values,
        dataset_seal_hash=original.dataset_seal_hash,
        materialization_hash=original.materialization_hash,
        lineage_hash=original.lineage_hash,
    )
    extra_target = BaseTargetRow(
        sample_id=extra.sample_id,
        daily_excess_return_path=(0.1, 0.2, 0.3, 0.4, 0.5),
        evaluation_time=extra.prediction_time,
        horizon_end_time=extra.prediction_time + timedelta(days=5),
        target_artifact_hash=SHA_D,
    )
    rows_with_extra = (*rows, extra)
    targets = _development_targets(plan, rows_with_extra, (*all_targets, extra_target))
    contract = BaseFeatureContract((_spec(),))
    _, predictions = BaseOOFRunner(
        feature_contract=contract,
        transformer_factory=lambda: RecordingTransformer(contract),
        trainer=RecordingTrainer(),
        output_dir=tmp_path,
        seed=7,
        config_hash=SHA_E,
        git_hash=SHA_F,
        audit_attestation_hash=SHA_B,
    ).run(
        rows=rows_with_extra,
        targets=targets,
        cycles=cycles,
        trading_calendar=calendar,
        plan=plan,
        synthetic=True,
    )
    predicted_ids = {row.sample_id for row in predictions}
    assert original.sample_id in predicted_ids
    assert extra.sample_id in predicted_ids


def test_runner_rejects_forbidden_targets_and_hash_drift(tmp_path: Path) -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = _plan(calendar, cycles)
    rows, all_targets = _rows_targets(calendar, cycles)
    contract = BaseFeatureContract((_spec(),))
    runner = BaseOOFRunner(
        feature_contract=contract,
        transformer_factory=lambda: RecordingTransformer(contract),
        trainer=RecordingTrainer(),
        output_dir=tmp_path,
        seed=7,
        config_hash=SHA_E,
        git_hash=SHA_F,
        audit_attestation_hash=SHA_B,
    )
    with pytest.raises(ValueError, match="forbidden target"):
        runner.run(
            rows=rows,
            targets=all_targets,
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=True,
        )

    drifted = list(rows)
    drifted[0] = OOFInputRow(
        sample_id=drifted[0].sample_id,
        cycle_id=drifted[0].cycle_id,
        stock_id=drifted[0].stock_id,
        cutoff_time=drifted[0].cutoff_time,
        prediction_time=drifted[0].prediction_time,
        feature_values=drifted[0].feature_values,
        dataset_seal_hash=SHA_B,
        materialization_hash=drifted[0].materialization_hash,
        lineage_hash=drifted[0].lineage_hash,
    )
    with pytest.raises(ValueError, match="dataset_seal_hash"):
        runner.run(
            rows=tuple(drifted),
            targets=_development_targets(plan, tuple(drifted), all_targets),
            cycles=cycles,
            trading_calendar=calendar,
            plan=plan,
            synthetic=True,
        )


def test_manifest_validator_rejects_tamper_and_incomplete_atomic_artifacts(tmp_path: Path) -> None:
    _, _, manifest, _, _, _ = _run_fixture(tmp_path)
    prediction_path = tmp_path / "base_oof_predictions.jsonl"
    manifest_path = tmp_path / "base_oof_manifest.json"

    validate_oof_artifacts(manifest_path=manifest_path, prediction_path=prediction_path)

    tampered_manifest = tmp_path / "tampered_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["dataset_seal_hash"] = SHA_B
    tampered_manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint"):
        validate_oof_artifacts(manifest_path=tampered_manifest, prediction_path=prediction_path)

    tampered_predictions = tmp_path / "tampered_predictions.jsonl"
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["daily_path"][0] = 999
    tampered_predictions.write_text(
        json.dumps(first, sort_keys=True) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prediction content"):
        validate_oof_artifacts(manifest_path=manifest_path, prediction_path=tampered_predictions)

    partial = tmp_path / "partial.jsonl.tmp"
    partial.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        load_jsonl_artifact(partial)

    with pytest.raises(FileExistsError):
        atomic_write_json(manifest_path, {"already": "exists"})
    with pytest.raises(FileExistsError):
        atomic_write_jsonl(prediction_path, [{"already": "exists"}])
    assert isinstance(manifest, OOFManifest)


def test_atomic_publish_is_no_clobber_and_streams_rows(tmp_path: Path, monkeypatch) -> None:
    final = tmp_path / "artifact.json"
    original_link = __import__("os").link

    def racing_link(src, dst):
        Path(dst).write_text("preoccupied", encoding="utf-8")
        original_link(src, dst)

    monkeypatch.setattr("os.link", racing_link)
    with pytest.raises(FileExistsError):
        atomic_write_json(final, {"new": True})
    assert final.read_text(encoding="utf-8") == "preoccupied"

    streamed = tmp_path / "streamed.jsonl"

    def bad_rows():
        yield {"row": 1}
        raise RuntimeError("stream interrupted")

    with pytest.raises(RuntimeError, match="stream interrupted"):
        atomic_write_jsonl(streamed, bad_rows())
    assert not streamed.exists()


def test_manifest_sensitivity_to_every_bound_hash() -> None:
    fold = FoldEvidence(
        fold_id=0,
        train_sample_id_hash=SHA_A,
        prediction_sample_id_hash=SHA_B,
        transform_fingerprint="sha256:" + SHA_C,
        checkpoint_sha256=SHA_D,
        prediction_content_hash=SHA_E,
        target_hash=SHA_F,
        cutoff_start=datetime(2026, 1, 1, 15, tzinfo=UTC),
        cutoff_end=datetime(2026, 1, 2, 15, tzinfo=UTC),
        prediction_count=1,
        train_start=datetime(2026, 1, 1, 15, tzinfo=UTC),
        train_end=datetime(2026, 1, 1, 15, tzinfo=UTC),
        validation_start=datetime(2026, 1, 2, 15, tzinfo=UTC),
        validation_end=datetime(2026, 1, 2, 15, tzinfo=UTC),
        test_start=datetime(2026, 1, 2, 15, tzinfo=UTC),
        test_end=datetime(2026, 1, 2, 15, tzinfo=UTC),
        purge_trading_days=5,
        embargo_trading_days=5,
        label_horizon_trading_days=5,
        fold_payload_hash=SHA_A,
    )
    row = OOFPredictionRow(
        sample_id="s1",
        cycle_id="c1",
        stock_id="stock-a",
        fold_id=0,
        cutoff_time=datetime(2026, 1, 1, 15, tzinfo=UTC),
        prediction_time=datetime(2026, 1, 2, 15, tzinfo=UTC),
        daily_path=(0.01, 0.02, 0.03, 0.04, 0.05),
        cumulative=(0.01, 0.06, 0.15),
        feature_schema_fingerprint="sha256:" + SHA_A,
        provenance="BASE_ONLY",
        classification="FORMAL_BASE_DAILY_PATH",
    )
    base = OOFManifest.create(
        classification="SYNTHETIC_NON_FORMAL",
        dataset_seal_hash=SHA_A,
        audit_attestation_hash=SHA_B,
        lineage_hash=SHA_C,
        materialization_hash=SHA_D,
        target_hash=SHA_E,
        split_hash=SHA_F,
        calendar_hash=SHA_A,
        config_hash=SHA_B,
        git_hash=SHA_C,
        feature_schema_hash="sha256:" + SHA_D,
        seed_hash=SHA_E,
        folds=(fold,),
        predictions=(row,),
        coverage={"predicted_count": 1},
    )
    for field_name in (
        "dataset_seal_hash",
        "audit_attestation_hash",
        "lineage_hash",
        "materialization_hash",
        "target_hash",
        "split_hash",
        "calendar_hash",
        "config_hash",
        "git_hash",
        "seed_hash",
    ):
        payload = base.without_fingerprint()
        payload[field_name] = SHA_F if getattr(base, field_name) != SHA_F else SHA_A
        changed = OOFManifest.create(**payload)
        assert changed.manifest_fingerprint != base.manifest_fingerprint


def test_validator_rejects_y_true_extra_schema_swapped_fold_hash_and_bad_coverage(
    tmp_path: Path,
) -> None:
    _, _, manifest, _, _, _ = _run_fixture(tmp_path)
    prediction_path = tmp_path / "base_oof_predictions.jsonl"
    manifest_path = tmp_path / "base_oof_manifest.json"
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["y_true"] = [0, 0, 0, 0, 0]
    y_true_path = tmp_path / "with_y_true.jsonl"
    y_true_path.write_text(
        json.dumps(first, sort_keys=True) + "\n" + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="y_true|schema|prediction content"):
        validate_oof_artifacts(manifest_path=manifest_path, prediction_path=y_true_path)

    if len(manifest.folds) < 2:
        pytest.fail("fixture must produce at least two folds")
    swapped_folds = list(manifest.folds)
    first_hash = swapped_folds[0].prediction_content_hash
    second_hash = swapped_folds[1].prediction_content_hash
    swapped_folds[0] = FoldEvidence(
        fold_id=swapped_folds[0].fold_id,
        train_sample_id_hash=swapped_folds[0].train_sample_id_hash,
        prediction_sample_id_hash=swapped_folds[0].prediction_sample_id_hash,
        transform_fingerprint=swapped_folds[0].transform_fingerprint,
        checkpoint_sha256=swapped_folds[0].checkpoint_sha256,
        prediction_content_hash=second_hash,
        target_hash=swapped_folds[0].target_hash,
        cutoff_start=swapped_folds[0].cutoff_start,
        cutoff_end=swapped_folds[0].cutoff_end,
        prediction_count=swapped_folds[0].prediction_count,
        train_start=swapped_folds[0].train_start,
        train_end=swapped_folds[0].train_end,
        validation_start=swapped_folds[0].validation_start,
        validation_end=swapped_folds[0].validation_end,
        test_start=swapped_folds[0].test_start,
        test_end=swapped_folds[0].test_end,
        purge_trading_days=swapped_folds[0].purge_trading_days,
        embargo_trading_days=swapped_folds[0].embargo_trading_days,
        label_horizon_trading_days=swapped_folds[0].label_horizon_trading_days,
        fold_payload_hash=swapped_folds[0].fold_payload_hash,
    )
    swapped_folds[1] = FoldEvidence(
        fold_id=swapped_folds[1].fold_id,
        train_sample_id_hash=swapped_folds[1].train_sample_id_hash,
        prediction_sample_id_hash=swapped_folds[1].prediction_sample_id_hash,
        transform_fingerprint=swapped_folds[1].transform_fingerprint,
        checkpoint_sha256=swapped_folds[1].checkpoint_sha256,
        prediction_content_hash=first_hash,
        target_hash=swapped_folds[1].target_hash,
        cutoff_start=swapped_folds[1].cutoff_start,
        cutoff_end=swapped_folds[1].cutoff_end,
        prediction_count=swapped_folds[1].prediction_count,
        train_start=swapped_folds[1].train_start,
        train_end=swapped_folds[1].train_end,
        validation_start=swapped_folds[1].validation_start,
        validation_end=swapped_folds[1].validation_end,
        test_start=swapped_folds[1].test_start,
        test_end=swapped_folds[1].test_end,
        purge_trading_days=swapped_folds[1].purge_trading_days,
        embargo_trading_days=swapped_folds[1].embargo_trading_days,
        label_horizon_trading_days=swapped_folds[1].label_horizon_trading_days,
        fold_payload_hash=swapped_folds[1].fold_payload_hash,
    )
    swapped = OOFManifest.create(
        **{
            **manifest.without_fingerprint(),
            "folds": tuple(swapped_folds),
            "prediction_content_hash": manifest.prediction_content_hash,
        }
    )
    swapped_path = tmp_path / "swapped_manifest.json"
    swapped_path.write_text(json.dumps(swapped.as_dict(), sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="fold prediction content"):
        validate_oof_artifacts(manifest_path=swapped_path, prediction_path=prediction_path)

    bad_coverage = OOFManifest.create(
        **{
            **manifest.without_fingerprint(),
            "coverage": {**dict(manifest.coverage), "eligible_count": 999},
            "prediction_content_hash": manifest.prediction_content_hash,
        }
    )
    bad_coverage_path = tmp_path / "bad_coverage_manifest.json"
    bad_coverage_path.write_text(json.dumps(bad_coverage.as_dict(), sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        validate_oof_artifacts(manifest_path=bad_coverage_path, prediction_path=prediction_path)
