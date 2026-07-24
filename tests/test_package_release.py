from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_release.ps1"


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class PackageReleaseScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.powershell = _powershell()
        if self.powershell is None:
            self.skipTest("PowerShell is not available")

    def test_fails_closed_when_phase_state_is_not_release_ready(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            status = root / "phase-state.json"
            blind = root / "blind.json"
            contract = root / "contract.md"
            submission = root / "submission.csv"
            output = root / "release"
            _write_json(
                status,
                {
                    "engineering_complete": True,
                    "data_ready": False,
                    "model_valid": False,
                    "backtest_valid": False,
                    "competition_ready": False,
                    "data_scope": "none",
                    "seat_detail_scope": "none",
                    "evidence_hashes": {},
                },
            )
            _write_json(blind, {"blind": True, "label_feedback_forbidden": True})
            contract.write_text("contract placeholder", encoding="utf-8")
            submission.write_text("ts_code,score\n000001.SZ,0.1\n", encoding="utf-8")

            completed = self._run_script(status, blind, contract, submission, output)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("data_ready must be true", completed.stderr)
            self.assertFalse((output / "release-manifest.json").exists())

    def test_generates_manifest_only_after_all_evidence_gates_pass(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            status = root / "phase-state.json"
            blind = root / "blind.json"
            contract = root / "contract.md"
            submission = root / "submission.csv"
            output = root / "release"
            _write_json(
                blind,
                {
                    "blind": True,
                    "label_feedback_forbidden": True,
                    "features": [{"path": "features.parquet", "sha256": "f" * 64}],
                    "labels": [{"path": "labels.parquet", "sha256": "a" * 64}],
                },
            )
            contract.write_text("competition contract placeholder", encoding="utf-8")
            submission.write_text("ts_code,score\n000001.SZ,0.1\n", encoding="utf-8")
            _write_json(
                status,
                {
                    "engineering_complete": True,
                    "data_ready": True,
                    "model_valid": True,
                    "backtest_valid": True,
                    "competition_ready": True,
                    "data_scope": "real",
                    "seat_detail_scope": "complete_seat",
                    "evidence_hashes": {
                        "blind_manifest_sha256": _sha256(blind),
                        "competition_contract_sha256": _sha256(contract),
                        "submission_sha256": _sha256(submission),
                    },
                },
            )

            completed = self._run_script(status, blind, contract, submission, output)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["release_gate"], "passed")
            self.assertEqual(manifest["phase_state"]["seat_detail_scope"], "complete_seat")
            self.assertEqual(
                manifest["output_boundary"],
                "manifest_only_no_raw_data_no_secrets_no_checkpoint_contents",
            )

    def test_release_ready_status_with_empty_hashes_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            status = root / "phase-state.json"
            blind = root / "blind.json"
            contract = root / "contract.md"
            submission = root / "submission.csv"
            output = root / "release"
            _write_json(
                blind,
                {
                    "blind": True,
                    "label_feedback_forbidden": True,
                    "features": [{"path": "features.parquet", "sha256": "f" * 64}],
                    "labels": [{"path": "labels.parquet", "sha256": "a" * 64}],
                },
            )
            contract.write_text("competition contract placeholder", encoding="utf-8")
            submission.write_text("ts_code,score\n000001.SZ,0.1\n", encoding="utf-8")
            _write_json(
                status,
                {
                    "engineering_complete": True,
                    "data_ready": True,
                    "model_valid": True,
                    "backtest_valid": True,
                    "competition_ready": True,
                    "data_scope": "real",
                    "seat_detail_scope": "complete_seat",
                    "evidence_hashes": {},
                },
            )

            completed = self._run_script(status, blind, contract, submission, output)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("evidence_hashes must be a non-empty map", completed.stderr)
            self.assertFalse((output / "release-manifest.json").exists())

    def _run_script(
        self,
        status: Path,
        blind: Path,
        contract: Path,
        submission: Path,
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert self.powershell is not None
        return subprocess.run(
            [
                self.powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-StatusFile",
                str(status),
                "-BlindManifest",
                str(blind),
                "-CompetitionContract",
                str(contract),
                "-SubmissionFile",
                str(submission),
                "-OutputDir",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
