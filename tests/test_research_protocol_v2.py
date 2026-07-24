from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.evaluation.protocol import (
    ExperimentProtocol,
    ProtocolWindow,
    freeze_protocol,
    load_experiment_protocol,
    verify_frozen_protocol,
)


class ResearchProtocolV2Test(unittest.TestCase):
    def test_production_v1_contract_uses_v2_flow_semantics_and_required_baselines(self) -> None:
        protocol = load_experiment_protocol("configs/experiments/production_v1.toml")

        self.assertEqual(protocol.protocol_id, "seatflow-production-v1-v2")
        self.assertEqual(protocol.seat_detail_scope, "institutional_proxy")
        self.assertEqual(protocol.model_family, "dt_flow_v2_unvalidated")
        self.assertEqual(
            protocol.baseline_models,
            ("logistic_regression", "lightgbm", "catboost", "mlp"),
        )
        self.assertFalse(protocol.window("future_blind").labels_may_be_read)

    def test_v2_protocol_marks_viewed_2026_as_contaminated_and_seals_future_labels(self) -> None:
        protocol = load_experiment_protocol("configs/experiments/research_protocol_v2.toml")

        self.assertEqual(protocol.protocol_id, "seatflow-institutional-proxy-v2")
        self.assertEqual(protocol.seat_detail_scope, "institutional_proxy")
        self.assertEqual(protocol.window("contaminated_oos").blind_status, "contaminated_oos")
        self.assertFalse(protocol.window("future_blind").labels_may_be_read)
        with self.assertRaisesRegex(ValueError, "labels are sealed"):
            protocol.assert_labels_readable("future_blind")

    def test_v2_protocol_round_trips_and_rejects_overlapping_windows(self) -> None:
        protocol = load_experiment_protocol("configs/experiments/research_protocol_v2.toml")
        with TemporaryDirectory() as temporary_directory:
            frozen = freeze_protocol(Path(temporary_directory), protocol)
            verified = verify_frozen_protocol(frozen.path)

        self.assertEqual(verified.protocol.window("future_blind").label_status, "unreleased")
        overlapping = ProtocolWindow(
            name="overlap",
            start=protocol.window("train").start,
            end=protocol.window("train").end,
            label_status="observed",
            blind_status="not_applicable",
        )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            ExperimentProtocol(
                protocol_id="overlap",
                data_scope="real",
                primary_metric="rank_ic",
                baseline_models=("naive",),
                seeds=(1,),
                hyperparameter_budget=0,
                max_horizon_days=5,
                purge_days=6,
                embargo_days=6,
                blind_start=protocol.blind_start,
                blind_end=protocol.blind_end,
                seat_detail_scope="institutional_proxy",
                label_status="mixed",
                blind_status="future_blind",
                model_family="event_logistic_baseline_v1",
                evaluation_windows=(protocol.window("train"), overlapping),
            )


if __name__ == "__main__":
    unittest.main()
