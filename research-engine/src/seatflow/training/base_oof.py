"""Strict fold-local Base OOF runner for Task 6C."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, Self, TypeVar

from seatflow.artifacts.atomic import atomic_write_json, atomic_write_jsonl
from seatflow.dataset.cycle import EventCycle
from seatflow.dataset.splits import WalkForwardPlan, validate_walk_forward_plan
from seatflow.models.base import BaseFeatureBatch, BaseFeatureContract, BasePrediction
from seatflow.training.base_oof_manifest import (
    FoldEvidence,
    OOFManifest,
    OOFPredictionRow,
    prediction_rows_hash,
    stable_hash,
)


@dataclass(frozen=True)
class OOFInputRow:
    """Fold-splittable Base OOF input row."""

    sample_id: str
    cycle_id: str
    stock_id: str
    cutoff_time: datetime
    prediction_time: datetime
    feature_values: Mapping[str, float]
    dataset_seal_hash: str
    materialization_hash: str
    lineage_hash: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "cycle_id", "stock_id"):
            if not getattr(self, name):
                raise ValueError(f"OOFInputRow {name} must be non-empty")
        _aware(self.cutoff_time, "cutoff_time")
        _aware(self.prediction_time, "prediction_time")
        if self.prediction_time <= self.cutoff_time:
            raise ValueError("OOFInputRow prediction_time must be after cutoff_time")
        for name in ("dataset_seal_hash", "materialization_hash", "lineage_hash"):
            _sha(getattr(self, name), name)
        frozen = MappingProxyType({str(k): float(v) for k, v in self.feature_values.items()})
        if not frozen:
            raise ValueError("OOFInputRow feature_values must be non-empty")
        if not all(math.isfinite(value) for value in frozen.values()):
            raise ValueError("OOFInputRow feature values must be finite")
        object.__setattr__(self, "feature_values", frozen)


@dataclass(frozen=True)
class BaseTargetRow:
    """Sealed target row joined to OOF inputs by sample_id only."""

    sample_id: str
    daily_excess_return_path: tuple[float, float, float, float, float]
    evaluation_time: datetime
    horizon_end_time: datetime
    target_artifact_hash: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("BaseTargetRow sample_id must be non-empty")
        _aware(self.evaluation_time, "evaluation_time")
        _aware(self.horizon_end_time, "horizon_end_time")
        if self.horizon_end_time < self.evaluation_time:
            raise ValueError("BaseTargetRow horizon_end_time precedes evaluation_time")
        _sha(self.target_artifact_hash, "target_artifact_hash")
        if (
            len(self.daily_excess_return_path) != 5
            or not all(math.isfinite(value) for value in self.daily_excess_return_path)
        ):
            raise ValueError("BaseTargetRow must contain five finite daily target values")


class FoldLocalTransformer(Protocol):
    fingerprint: str

    def fit(self, rows: tuple[OOFInputRow, ...]) -> Self: ...

    def transform(self, rows: tuple[OOFInputRow, ...]) -> BaseFeatureBatch: ...


class Predictor(Protocol):
    def predict(
        self,
        batch: BaseFeatureBatch,
        *,
        rows: tuple[OOFInputRow, ...],
    ) -> tuple[BasePrediction, ...]: ...


@dataclass(frozen=True)
class FoldTrainingBundle:
    fold_id: int
    train_sample_ids: tuple[str, ...]
    transform_fingerprint: str
    checkpoint_sha256: str
    predictor: Predictor
    seed: int

    def __post_init__(self) -> None:
        _sha_uri(self.transform_fingerprint, "transform_fingerprint")
        _sha(self.checkpoint_sha256, "checkpoint_sha256")
        if not self.train_sample_ids:
            raise ValueError("FoldTrainingBundle train_sample_ids must be non-empty")


class BaseOOFTrainer(Protocol):
    def fit(
        self,
        *,
        fold_id: int,
        train_batch: BaseFeatureBatch,
        train_targets: tuple[BaseTargetRow, ...],
        train_sample_ids: tuple[str, ...],
        transform_fingerprint: str,
        seed: int,
    ) -> FoldTrainingBundle: ...


@dataclass(frozen=True)
class BaseOOFRunner:
    """Run strict expanding OOF with injected fold-local components."""

    feature_contract: BaseFeatureContract
    transformer_factory: Callable[[], FoldLocalTransformer]
    trainer: BaseOOFTrainer
    output_dir: Path
    seed: int
    config_hash: str
    git_hash: str
    audit_attestation_hash: str

    def __post_init__(self) -> None:
        _sha(self.config_hash, "config_hash")
        _sha(self.git_hash, "git_hash")
        _sha(self.audit_attestation_hash, "audit_attestation_hash")

    def run(
        self,
        *,
        rows: Sequence[OOFInputRow],
        targets: Sequence[BaseTargetRow],
        cycles: Sequence[EventCycle],
        trading_calendar: Sequence[datetime],
        plan: WalkForwardPlan,
        synthetic: bool,
    ) -> tuple[OOFManifest, tuple[OOFPredictionRow, ...]]:
        if not synthetic:
            raise ValueError(
                "FORMAL_AUDIT_ATTESTATION_REQUIRED: Task 6C formal runner is blocked "
                "until issued Task5E seal/attestation adapters are integrated"
            )
        validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=trading_calendar)
        row_by_sample = _unique_by_sample(tuple(rows), "rows")
        target_by_sample = _unique_by_sample(tuple(targets), "targets")
        self._validate_row_hashes(tuple(row_by_sample.values()))
        self._validate_target_hashes(tuple(target_by_sample.values()))
        cycle_ids = {cycle.cycle_id for cycle in cycles}
        unknown_cycles = sorted({row.cycle_id for row in row_by_sample.values()} - cycle_ids)
        if unknown_cycles:
            raise ValueError(f"OOF rows reference outside cycles: {unknown_cycles}")
        rows_by_cycle = _rows_by_cycle(tuple(row_by_sample.values()))
        development_sample_ids = _development_sample_ids(rows_by_cycle, plan)
        missing_targets = sorted(development_sample_ids - set(target_by_sample))
        forbidden_targets = sorted(set(target_by_sample) - development_sample_ids)
        if missing_targets:
            raise ValueError(f"target sample ids missing for development rows: {missing_targets}")
        if forbidden_targets:
            raise ValueError(f"forbidden target sample ids outside development folds: {forbidden_targets}")

        predictions: list[OOFPredictionRow] = []
        fold_evidence: list[FoldEvidence] = []
        expected_prediction_ids: set[str] = set()
        for fold in plan.folds:
            train_rows = _rows_for_cycles(rows_by_cycle, fold.train_cycle_ids)
            prediction_rows = _rows_for_cycles(rows_by_cycle, fold.prediction_cycle_ids)
            expected_prediction_ids.update(row.sample_id for row in prediction_rows)
            transformer = self.transformer_factory().fit(train_rows)
            _sha_uri(transformer.fingerprint, "transform_fingerprint")
            train_batch = transformer.transform(train_rows)
            self._require_exact_batch(train_batch)
            train_sample_ids = tuple(row.sample_id for row in train_rows)
            train_targets = tuple(target_by_sample[sample_id] for sample_id in train_sample_ids)
            bundle = self.trainer.fit(
                fold_id=fold.fold_id,
                train_batch=train_batch,
                train_targets=train_targets,
                train_sample_ids=train_sample_ids,
                transform_fingerprint=transformer.fingerprint,
                seed=self.seed,
            )
            self._validate_bundle(bundle, fold.fold_id, train_sample_ids, transformer.fingerprint)
            prediction_batch = transformer.transform(prediction_rows)
            self._require_exact_batch(prediction_batch)
            base_predictions = bundle.predictor.predict(prediction_batch, rows=prediction_rows)
            fold_predictions = self._prediction_rows(
                fold_id=fold.fold_id,
                source_rows=prediction_rows,
                predictions=base_predictions,
            )
            predictions.extend(fold_predictions)
            fold_evidence.append(
                FoldEvidence(
                    fold_id=fold.fold_id,
                    train_sample_id_hash=stable_hash(train_sample_ids),
                    prediction_sample_id_hash=stable_hash(tuple(row.sample_id for row in prediction_rows)),
                    transform_fingerprint=transformer.fingerprint,
                    checkpoint_sha256=bundle.checkpoint_sha256,
                    prediction_content_hash=prediction_rows_hash(fold_predictions),
                    target_hash=stable_hash(
                        tuple(
                            {
                                "sample_id": target.sample_id,
                                "daily_excess_return_path": target.daily_excess_return_path,
                                "target_artifact_hash": target.target_artifact_hash,
                            }
                            for target in train_targets
                        )
                    ),
                    cutoff_start=fold.train_start,
                    cutoff_end=fold.test_end,
                    prediction_count=len(fold_predictions),
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    validation_start=fold.validation_start,
                    validation_end=fold.validation_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    purge_trading_days=fold.purge_trading_days,
                    embargo_trading_days=fold.embargo_trading_days,
                    label_horizon_trading_days=fold.label_horizon_trading_days,
                    fold_payload_hash=stable_hash(fold.as_dict()),
                )
            )

        prediction_tuple = tuple(sorted(predictions, key=lambda row: (row.sample_id, row.fold_id)))
        _assert_exactly_once(prediction_tuple, expected_prediction_ids)
        classification = (
            OOFManifest.SYNTHETIC_CLASSIFICATION if synthetic else "FORMAL"
        )
        first_row = next(iter(row_by_sample.values()))
        first_target = next(iter(target_by_sample.values()))
        manifest = OOFManifest.create(
            classification=classification,
            dataset_seal_hash=first_row.dataset_seal_hash,
            audit_attestation_hash=self.audit_attestation_hash,
            lineage_hash=first_row.lineage_hash,
            materialization_hash=first_row.materialization_hash,
            target_hash=first_target.target_artifact_hash,
            split_hash=plan.fingerprint,
            calendar_hash=plan.calendar_fingerprint,
            config_hash=self.config_hash,
            git_hash=self.git_hash,
            feature_schema_hash=self.feature_contract.schema_fingerprint,
            seed_hash=stable_hash({"seed": self.seed}),
            folds=tuple(fold_evidence),
            predictions=prediction_tuple,
            coverage={
                "input_count": len(row_by_sample),
                "predicted_count": len(prediction_tuple),
                "fold_count": len(plan.folds),
                "eligible_count": len(expected_prediction_ids),
            },
        )
        self._publish(prediction_tuple, manifest)
        return manifest, prediction_tuple

    def _require_exact_batch(self, batch: BaseFeatureBatch) -> None:
        try:
            self.feature_contract.revalidate_batch(batch)
        except ValueError as exc:
            raise ValueError(
                "transform output must be issued by the runner's exact BaseFeatureContract"
            ) from exc

    def _validate_bundle(
        self,
        bundle: FoldTrainingBundle,
        fold_id: int,
        train_sample_ids: tuple[str, ...],
        transform_fingerprint: str,
    ) -> None:
        if bundle.fold_id != fold_id:
            raise ValueError("trainer bundle fold_id mismatch")
        if bundle.train_sample_ids != train_sample_ids:
            raise ValueError("trainer bundle train sample hash binding mismatch")
        if bundle.transform_fingerprint != transform_fingerprint:
            raise ValueError("trainer bundle transform fingerprint mismatch")
        if bundle.seed != self.seed:
            raise ValueError("trainer bundle seed mismatch")

    @staticmethod
    def _validate_row_hashes(rows: tuple[OOFInputRow, ...]) -> None:
        for field_name in ("dataset_seal_hash", "materialization_hash", "lineage_hash"):
            values = {getattr(row, field_name) for row in rows}
            if len(values) != 1:
                raise ValueError(f"{field_name} must be identical for all OOF rows")

    @staticmethod
    def _validate_target_hashes(targets: tuple[BaseTargetRow, ...]) -> None:
        values = {target.target_artifact_hash for target in targets}
        if len(values) != 1:
            raise ValueError("target_artifact_hash must be identical for all target rows")

    def _prediction_rows(
        self,
        *,
        fold_id: int,
        source_rows: tuple[OOFInputRow, ...],
        predictions: tuple[BasePrediction, ...],
    ) -> tuple[OOFPredictionRow, ...]:
        if len(source_rows) != len(predictions):
            raise ValueError("predictor returned missing or extra rows")
        out: list[OOFPredictionRow] = []
        for row, prediction in zip(source_rows, predictions):
            if prediction.stock_id != row.stock_id:
                raise ValueError("predictor stock_id mismatch")
            if prediction.cutoff_time != row.cutoff_time:
                raise ValueError("predictor cutoff_time mismatch")
            if prediction.prediction_time != row.prediction_time:
                raise ValueError("predictor prediction_time mismatch")
            if prediction.feature_schema_fingerprint != self.feature_contract.schema_fingerprint:
                raise ValueError("predictor feature schema mismatch")
            if prediction.cumulative is None:
                raise ValueError("predictor cumulative path is missing")
            out.append(
                OOFPredictionRow(
                    sample_id=row.sample_id,
                    cycle_id=row.cycle_id,
                    stock_id=row.stock_id,
                    fold_id=fold_id,
                    cutoff_time=row.cutoff_time,
                    prediction_time=row.prediction_time,
                    daily_path=prediction.daily_path,
                    cumulative=prediction.cumulative,
                    feature_schema_fingerprint=prediction.feature_schema_fingerprint,
                    provenance=prediction.provenance,
                    classification=prediction.classification,
                )
            )
        return tuple(out)

    def _publish(
        self,
        predictions: tuple[OOFPredictionRow, ...],
        manifest: OOFManifest,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(
            self.output_dir / "base_oof_predictions.jsonl",
            (row.as_dict() for row in predictions),
        )
        atomic_write_json(self.output_dir / "base_oof_manifest.json", manifest.as_dict())


def _rows_for_cycles(
    rows_by_cycle: Mapping[str, tuple[OOFInputRow, ...]],
    cycle_ids: frozenset[str],
) -> tuple[OOFInputRow, ...]:
    missing = sorted(cycle_id for cycle_id in cycle_ids if cycle_id not in rows_by_cycle)
    if missing:
        raise ValueError(f"missing OOF rows for cycles: {missing}")
    rows: list[OOFInputRow] = []
    for cycle_id in sorted(cycle_ids):
        rows.extend(rows_by_cycle[cycle_id])
    return tuple(rows)


def _rows_by_cycle(rows: tuple[OOFInputRow, ...]) -> dict[str, tuple[OOFInputRow, ...]]:
    grouped: dict[str, list[OOFInputRow]] = {}
    for row in rows:
        grouped.setdefault(row.cycle_id, []).append(row)
    return {
        cycle_id: tuple(sorted(group_rows, key=lambda row: row.sample_id))
        for cycle_id, group_rows in grouped.items()
    }


def _development_sample_ids(
    rows_by_cycle: Mapping[str, tuple[OOFInputRow, ...]],
    plan: WalkForwardPlan,
) -> set[str]:
    development_cycle_ids: set[str] = set()
    for fold in plan.folds:
        development_cycle_ids.update(fold.train_cycle_ids)
        development_cycle_ids.update(fold.prediction_cycle_ids)
    ids: set[str] = set()
    for cycle_id in development_cycle_ids:
        for row in rows_by_cycle.get(cycle_id, ()):
            ids.add(row.sample_id)
    return ids


_SampleRow = TypeVar("_SampleRow", OOFInputRow, BaseTargetRow)


def _unique_by_sample(
    rows: tuple[_SampleRow, ...],
    name: str,
) -> dict[str, _SampleRow]:
    by_id: dict[str, _SampleRow] = {}
    for row in rows:
        if row.sample_id in by_id:
            raise ValueError(f"duplicate sample_id in {name}: {row.sample_id}")
        by_id[row.sample_id] = row
    if not by_id:
        raise ValueError(f"{name} must be non-empty")
    return by_id


def _assert_exactly_once(
    predictions: tuple[OOFPredictionRow, ...],
    expected_sample_ids: set[str],
) -> None:
    counts: dict[str, int] = {}
    for prediction in predictions:
        counts[prediction.sample_id] = counts.get(prediction.sample_id, 0) + 1
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count != 1)
    missing = sorted(expected_sample_ids - set(counts))
    extra = sorted(set(counts) - expected_sample_ids)
    if duplicates or missing or extra:
        raise ValueError(
            f"OOF exactly-once violation duplicates={duplicates} missing={missing} extra={extra}"
        )


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 64-character sha256 hex fingerprint")


def _sha_uri(value: str, name: str) -> None:
    if not value.startswith("sha256:"):
        raise ValueError(f"{name} must be sha256:<64 hex>")
    _sha(value.removeprefix("sha256:"), name)
