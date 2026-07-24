from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.cli import main
from seatflow.status import new_phase_status, write_phase_status


class PromotionCliTest(unittest.TestCase):
    def test_negative_metrics_write_decision_and_keep_phase_fail_closed(self) -> None:
        raw_metrics = {
            "audited": False,
            "metrics": [],
            "ablations": [],
            "final_blind_manifest": {"split_status": "contaminated_oos"},
        }
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metrics_path = root / "metrics.json"
            decision_path = root / "promotion.json"
            status_path = root / "phase-state.json"
            metrics_path.write_text(json.dumps(raw_metrics), encoding="utf-8-sig")
            write_phase_status(
                status_path,
                new_phase_status(
                    engineering_complete=True,
                    data_ready=False,
                    model_valid=False,
                    backtest_valid=False,
                    competition_ready=False,
                    scope="institutional_only",
                    data_scope="real",
                    evidence_hashes={},
                ),
            )

            main(
                [
                    "evaluate-promotion",
                    "--metrics-file",
                    str(metrics_path),
                    "--output",
                    str(decision_path),
                    "--status-file",
                    str(status_path),
                ]
            )
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertTrue(decision["negative_result"])
        self.assertFalse(status["model_valid"])
        self.assertFalse(status["backtest_valid"])
        self.assertFalse(status["competition_ready"])
        self.assertIn("promotion_decision", status["evidence_hashes"])


if __name__ == "__main__":
    unittest.main()
