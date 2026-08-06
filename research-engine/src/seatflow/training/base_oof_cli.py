"""Machine-readable CLI for Task 6C Base OOF smoke/formal gates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import torch

from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.splits import WalkForwardConfig, create_walk_forward_plan
from seatflow.features.registry import FeatureScope, FeatureSpec, SourceKind
from seatflow.models.base import BaseFeatureBatch, BaseFeatureContract, BasePrediction
from seatflow.training.base_oof import BaseOOFRunner, BaseTargetRow, FoldTrainingBundle, OOFInputRow
from seatflow.training.base_oof_manifest import OOFManifest, OOFPredictionRow


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="base-oof")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.formal:
            return _emit(
                2,
                {
                    "status": "BLOCKED",
                    "classification": "FORMAL_BLOCKED",
                    "reasons": [
                        "FORMAL_DATASET_SEAL_REQUIRED",
                        "FORMAL_AUDIT_ATTESTATION_REQUIRED",
                        "FORMAL_TARGET_ARTIFACT_REQUIRED",
                        "REAL_BASE_TRAINER_REQUIRED",
                    ],
                },
            )
        if args.synthetic_smoke:
            if args.output_dir is None:
                raise ValueError("output_dir is required for synthetic smoke")
            manifest, predictions = _run_synthetic_smoke(args.output_dir)
            return _emit(
                0,
                {
                    "status": "COMPLETED",
                    "classification": manifest.classification,
                    "manifest_path": str(args.output_dir / "base_oof_manifest.json"),
                    "predictions_path": str(args.output_dir / "base_oof_predictions.jsonl"),
                    "coverage": dict(manifest.coverage),
                    "prediction_count": len(predictions),
                    "manifest_fingerprint": manifest.manifest_fingerprint,
                },
            )
        raise ValueError("either --synthetic-smoke or --formal is required")
    except Exception as exc:
        return _emit(1, {"status": "ERROR", "reason": str(exc)})


def _emit(exit_code: int, payload: dict[str, object]) -> int:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return exit_code


def _run_synthetic_smoke(output_dir: Path) -> tuple[OOFManifest, tuple[OOFPredictionRow, ...]]:
    calendar = tuple(datetime(2026, 1, 1, 15, tzinfo=UTC) + timedelta(days=i) for i in range(85))
    cycles = tuple(
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
    plan = create_walk_forward_plan(
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
    rows: list[OOFInputRow] = []
    targets: list[BaseTargetRow] = []
    for idx, cycle in enumerate(cycles):
        cutoff = cycle.cycle_end
        if cutoff is None:
            raise ValueError("synthetic cycle unexpectedly open")
        rows.append(
            OOFInputRow(
                sample_id=f"sample-{idx:03d}",
                cycle_id=cycle.cycle_id,
                stock_id=cycle.stock_id,
                cutoff_time=cutoff,
                prediction_time=calendar[idx + 1],
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
                evaluation_time=calendar[idx + 1],
                horizon_end_time=calendar[idx + 5],
                target_artifact_hash=SHA_D,
            )
        )
    contract = BaseFeatureContract(
        (
            FeatureSpec(
                name="stock_volatility_5d",
                version="1.0",
                horizons=[5],
                dependencies=[],
                description="synthetic fixture",
                scope=FeatureScope.STOCK_PIT,
                source=SourceKind.STOCK_PIT,
            ),
        )
    )
    runner = BaseOOFRunner(
        feature_contract=contract,
        transformer_factory=lambda: _SyntheticTransformer(contract),
        trainer=_SyntheticTrainer(),
        output_dir=output_dir,
        seed=7,
        config_hash=SHA_E,
        git_hash=SHA_F,
        audit_attestation_hash=SHA_B,
    )
    development_cycle_ids: set[str] = set()
    for fold in plan.folds:
        development_cycle_ids.update(fold.train_cycle_ids)
        development_cycle_ids.update(fold.prediction_cycle_ids)
    development_sample_ids = {
        row.sample_id for row in rows if row.cycle_id in development_cycle_ids
    }
    development_targets = tuple(
        target for target in targets if target.sample_id in development_sample_ids
    )
    return runner.run(
        rows=tuple(rows),
        targets=development_targets,
        cycles=cycles,
        trading_calendar=calendar,
        plan=plan,
        synthetic=True,
    )


class _SyntheticTransformer:
    def __init__(self, contract: BaseFeatureContract) -> None:
        self.contract = contract
        self.mean = 0.0
        self.fingerprint = "sha256:" + SHA_E

    def fit(self, rows: tuple[OOFInputRow, ...]) -> "_SyntheticTransformer":
        self.mean = sum(row.feature_values["stock_volatility_5d"] for row in rows) / len(rows)
        return self

    def transform(self, rows: tuple[OOFInputRow, ...]) -> BaseFeatureBatch:
        return self.contract.validate_batch(
            torch.tensor(
                [[row.feature_values["stock_volatility_5d"] - self.mean] for row in rows],
                dtype=torch.float32,
            ),
            [row.stock_id for row in rows],
            feature_names=self.contract.feature_names,
        )


class _SyntheticTrainer:
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
        return FoldTrainingBundle(
            fold_id=fold_id,
            train_sample_ids=train_sample_ids,
            transform_fingerprint=transform_fingerprint,
            checkpoint_sha256=SHA_F,
            predictor=_SyntheticPredictor(),
            seed=seed,
        )


class _SyntheticPredictor:
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


if __name__ == "__main__":
    raise SystemExit(main())
