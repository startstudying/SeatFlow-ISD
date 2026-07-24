import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.status import (
    default_phase_status,
    load_phase_status,
    new_phase_status,
    phase_status_from_promotion,
    write_phase_status,
)


class PhaseStatusTest(unittest.TestCase):
    def test_missing_status_is_explicitly_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            status = load_phase_status(Path(temporary_directory) / "phase-state.json")

        self.assertFalse(status.data_ready)
        self.assertEqual(status.scope, "none")
        self.assertFalse(status.status_file_present)

    def test_status_rejects_invalid_claim_progression(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_valid requires data_ready"):
            new_phase_status(
                engineering_complete=True,
                data_ready=False,
                model_valid=True,
                backtest_valid=False,
                competition_ready=False,
                scope="institutional_only",
                evidence_hashes={},
            )

    def test_status_round_trip(self) -> None:
        status = new_phase_status(
            engineering_complete=True,
            data_ready=True,
            model_valid=False,
            backtest_valid=False,
            competition_ready=False,
            scope="institutional_only",
            evidence_hashes={"audit": "a" * 64},
            blocked_reasons=("Blind evidence is pending.",),
        )
        with TemporaryDirectory() as temporary_directory:
            path = write_phase_status(Path(temporary_directory) / "phase-state.json", status)
            loaded = load_phase_status(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(loaded.data_ready)
        self.assertTrue(loaded.status_file_present)
        self.assertEqual(payload["scope"], "institutional_only")

    def test_default_status_has_no_implicit_evidence(self) -> None:
        self.assertEqual(default_phase_status().evidence_hashes, {})

    def test_negative_promotion_decision_cannot_claim_validity(self) -> None:
        promotion_decision = {
            "promoted": False,
            "blocked_reasons": ["auc_lift_below_threshold"],
        }

        status = phase_status_from_promotion(
            engineering_complete=True,
            data_ready=True,
            scope="complete_seat",
            evidence_hashes={"promotion": "b" * 64},
            promotion_decision=promotion_decision,
        )

        self.assertFalse(status.model_valid)
        self.assertFalse(status.backtest_valid)
        self.assertFalse(status.competition_ready)
        self.assertIn("auc_lift_below_threshold", status.blocked_reasons)

    def test_phase_status_uses_explicit_promotion_validity_flags(self) -> None:
        status = phase_status_from_promotion(
            engineering_complete=True,
            data_ready=True,
            scope="complete_seat",
            evidence_hashes={"promotion": "e" * 64},
            promotion_decision={
                "promoted": True,
                "model_valid": True,
                "backtest_valid": True,
                "competition_ready": False,
                "blocked_reasons": [],
            },
        )

        self.assertTrue(status.model_valid)
        self.assertTrue(status.backtest_valid)
        self.assertFalse(status.competition_ready)

    def test_load_rejects_validity_with_failed_promotion_payload(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "phase-state.json"
            path.write_text(
                json.dumps(
                    {
                        "engineering_complete": True,
                        "data_ready": True,
                        "model_valid": True,
                        "backtest_valid": True,
                        "competition_ready": True,
                        "scope": "complete_seat",
                        "data_scope": "real",
                        "evidence_hashes": {"promotion": "c" * 64},
                        "blocked_reasons": [],
                        "promotion_decision": {"promoted": False},
                    }
                ),
                encoding="utf-8",
            )

            status = load_phase_status(path)

        self.assertFalse(status.model_valid)
        self.assertIn("validity claims require promoted", status.blocked_reasons[0])

    def test_promoted_metrics_cannot_override_data_readiness(self) -> None:
        status = phase_status_from_promotion(
            engineering_complete=True,
            data_ready=False,
            scope="institutional_only",
            evidence_hashes={"promotion": "d" * 64},
            promotion_decision={"promoted": True, "competition_ready": True},
        )

        self.assertFalse(status.model_valid)
        self.assertFalse(status.backtest_valid)
        self.assertFalse(status.competition_ready)
        self.assertIn("promotion_model_valid_not_true", status.blocked_reasons)
        self.assertIn("promotion_backtest_valid_not_true", status.blocked_reasons)

    def test_promoted_without_model_or_backtest_flags_fails_closed(self) -> None:
        status = phase_status_from_promotion(
            engineering_complete=True,
            data_ready=True,
            scope="complete_seat",
            evidence_hashes={"promotion": "f" * 64},
            promotion_decision={
                "promoted": True,
                "model_valid": True,
                "backtest_valid": False,
                "competition_ready": True,
            },
        )

        self.assertFalse(status.model_valid)
        self.assertFalse(status.backtest_valid)
        self.assertFalse(status.competition_ready)
        self.assertIn("promotion_backtest_valid_not_true", status.blocked_reasons)
        self.assertIn(
            "promotion_competition_ready_without_backtest_valid",
            status.blocked_reasons,
        )
