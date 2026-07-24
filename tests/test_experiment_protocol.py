import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.evaluation.protocol import (
    ExperimentProtocol,
    freeze_protocol,
    load_experiment_protocol,
    verify_frozen_protocol,
)


class ExperimentProtocolTest(unittest.TestCase):
    def test_protocol_rejects_embargo_shorter_than_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "embargo_days"):
            ExperimentProtocol(
                protocol_id="invalid",
                data_scope="synthetic",
                primary_metric="rank_ic",
                baseline_models=("naive",),
                seeds=(1,),
                hyperparameter_budget=0,
                max_horizon_days=5,
                purge_days=5,
                embargo_days=4,
                blind_start=date(2025, 1, 1),
                blind_end=date(2025, 1, 2),
            )

    def test_freeze_and_verify_protocol_lock(self) -> None:
        protocol = load_experiment_protocol("configs/experiments/protocol.toml")
        with TemporaryDirectory() as temporary_directory:
            frozen = freeze_protocol(Path(temporary_directory), protocol)
            verified = verify_frozen_protocol(frozen.path)
            repeated = freeze_protocol(Path(temporary_directory), protocol)

        self.assertEqual(verified.sha256, frozen.sha256)
        self.assertEqual(repeated.sha256, frozen.sha256)
        self.assertEqual(verified.protocol.protocol_id, "engineering-smoke-v1")

    def test_protocol_id_cannot_be_reused_with_changed_content(self) -> None:
        protocol = load_experiment_protocol("configs/experiments/protocol.toml")
        changed = ExperimentProtocol(
            protocol_id=protocol.protocol_id,
            data_scope=protocol.data_scope,
            primary_metric="auc",
            baseline_models=protocol.baseline_models,
            seeds=protocol.seeds,
            hyperparameter_budget=protocol.hyperparameter_budget,
            max_horizon_days=protocol.max_horizon_days,
            purge_days=protocol.purge_days,
            embargo_days=protocol.embargo_days,
            blind_start=protocol.blind_start,
            blind_end=protocol.blind_end,
        )
        with TemporaryDirectory() as temporary_directory:
            freeze_protocol(Path(temporary_directory), protocol)
            with self.assertRaisesRegex(ValueError, "already frozen"):
                freeze_protocol(Path(temporary_directory), changed)
