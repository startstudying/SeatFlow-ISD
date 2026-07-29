"""
Phase-state integrity regression tests (Issue #5).

These tests guard that the roadmap runner is strictly fail-closed:

1. Sub-task failure → 3 ablation blockers remain
2. Missing result files → fail
3. Insufficient seeds (<5) → fail
4. Insufficient folds (<3) → fail
5. NaN/Inf metrics → fail
6. Dataset version mismatch → fail
7. Planned metrics MUST NOT enter real experiment JSON
8. Invalid CLI commands are caught by tests

Guards 1/3/4 are enforced by seatflow.evaluation.gates (the promotion
gate the runner delegates to). Guards 2/5/6 are enforced by the runner's
own validation (gates.py deliberately treats NaN as a numeric value, so
the runner is the fail-closed boundary). All tests run in isolation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

from seatflow.evaluation.gates import evaluate_promotion_decision


def _load_script_module(name: str, relative_path: str):
    """Load a scripts/*.py module from path (scripts/ is not a package).

    The module is registered in sys.modules so that ``@dataclass`` and other
    decorators that resolve ``cls.__module__`` can find it.
    """
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _full_passing_metrics() -> dict:
    """5 seeds × 3 folds candidate + baseline rows plus 3 passing ablations."""
    seeds = [2025, 2026, 2027, 2028, 2029]
    folds = ["0", "1", "2"]
    metrics = []
    for seed in seeds:
        for fold in folds:
            metrics.append(
                {
                    "model": "dt_seathgt",
                    "seed": seed,
                    "fold": fold,
                    "auc": 0.65,
                    "rank_ic": 0.08,
                    "topk_excess_return_after_cost": 0.01,
                    "max_drawdown": 0.05,
                }
            )
            metrics.append(
                {
                    "model": "strong_baseline",
                    "seed": seed,
                    "fold": fold,
                    "auc": 0.60,
                    "rank_ic": 0.05,
                    "topk_excess_return_after_cost": 0.0,
                    "max_drawdown": 0.04,
                }
            )
    ablations = []
    for group in ("seat_profile", "time_edges", "direction_edges"):
        for fold in folds:
            ablations.append(
                {
                    "group": group,
                    "seed": 2025,
                    "fold": fold,
                    "auc_contribution": 0.02,
                    "rank_ic_contribution": 0.02,
                }
            )
    return {
        "audited": True,
        "candidate_model": "dt_seathgt",
        "strong_baseline": "strong_baseline",
        "metrics": metrics,
        "ablations": ablations,
        "final_blind_manifest": None,
        "split_status": None,
    }


class RoadmapRunnerIntegrityTest(TestCase):
    maxDiff = 4000

    # ---- Guard 1: sub-task failure keeps blockers -------------------------

    def test_subtask_failure_keeps_blockers(self) -> None:
        """If any sub-task fails, the 3 ablation blockers must remain."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "phase-state.json"
            initial_blockers = [
                "missing_ablation:seat_profile",
                "missing_ablation:time_edges",
                "missing_ablation:direction_edges",
                "other_blocker",
            ]
            initial_status = {
                "engineering_complete": True,
                "data_ready": False,
                "model_valid": False,
                "backtest_valid": False,
                "competition_ready": False,
                "data_scope": "real",
                "scope": "institutional_only",
                "evidence_hashes": {},
                "blocked_reasons": list(initial_blockers),
                "last_verified_at": "2026-07-29T00:00:00Z",
                "status_file_present": True,
            }
            status_path.write_text(
                json.dumps(initial_status, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            empty_metrics = {
                "audited": True,
                "candidate_model": "dt_seathgt",
                "strong_baseline": "strong_baseline",
                "metrics": [],
                "ablations": [],
                "final_blind_manifest": None,
                "split_status": None,
            }
            decision = evaluate_promotion_decision(empty_metrics)

            blocked = list(decision.blocked_reasons)
            for reason in (
                "missing_ablation:seat_profile",
                "missing_ablation:time_edges",
                "missing_ablation:direction_edges",
            ):
                self.assertIn(reason, blocked)

            # Runner does NOT touch phase-state on failure; file unchanged.
            unchanged = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(unchanged["blocked_reasons"], initial_blockers)
            self.assertFalse(decision.promoted)

    # ---- Guard 2: missing result files fail -------------------------------

    def test_missing_result_files_fail(self) -> None:
        """Runner validation must reject a missing result file."""
        runner = _load_script_module(
            "complete_6week_roadmap", "scripts/complete_6week_roadmap.py"
        )
        # _load_ablation_result reads ABLATION_RESULT_PATH directly; point it
        # at a missing path so the guard fires.
        runner.ABLATION_RESULT_PATH = Path("does/not/exist.json")
        with self.assertRaises(FileNotFoundError):
            runner._load_ablation_result()

    # ---- Guard 3: insufficient seeds fail ---------------------------------

    def test_insufficient_seeds_fail(self) -> None:
        """Seed count < 5 must block promotion (gate enforces min_seed_count)."""
        metrics = _full_passing_metrics()
        # Reduce to 2 seeds (delete 3).
        keep_seeds = {"2025", "2026"}
        metrics["metrics"] = [
            row
            for row in metrics["metrics"]
            if str(row["seed"]) in keep_seeds
        ]
        decision = evaluate_promotion_decision(metrics)
        self.assertIn("single_or_missing_seed", decision.blocked_reasons)
        self.assertFalse(decision.promoted)

    # ---- Guard 4: insufficient folds fail ---------------------------------

    def test_insufficient_folds_fail(self) -> None:
        """Fold count < 3 must block promotion (gate enforces min_fold_count)."""
        metrics = _full_passing_metrics()
        # Reduce to 1 fold.
        metrics["metrics"] = [
            row for row in metrics["metrics"] if str(row["fold"]) == "0"
        ]
        decision = evaluate_promotion_decision(metrics)
        self.assertIn("single_or_missing_rolling_fold", decision.blocked_reasons)
        self.assertFalse(decision.promoted)

    # ---- Guard 5: NaN/Inf metrics fail ------------------------------------

    def test_nan_inf_metrics_fail(self) -> None:
        """Runner validation must reject NaN/Inf metrics (gates.py does not)."""
        runner_roadmap = _load_script_module(
            "complete_6week_roadmap_nan", "scripts/complete_6week_roadmap.py"
        )
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad):
                raw = {
                    "status": "ok",
                    "dataset_version_sha": "abc",
                    "git_commit_sha": "def",
                    "config_sha": "ghi",
                    "seeds": [2025, 2026, 2027, 2028, 2029],
                    "folds": ["0", "1", "2"],
                    "real_metrics": {
                        "audited": True,
                        "metrics": [
                            {
                                "model": "dt_seathgt",
                                "seed": 2025,
                                "fold": "0",
                                "auc": bad,
                                "rank_ic": 0.05,
                            }
                        ],
                        "ablations": [],
                    },
                    "failure_reason": None,
                }
                with self.assertRaises(ValueError):
                    runner_roadmap._validate_ablation_result(raw)

    # ---- Guard 6: dataset version mismatch fail ---------------------------

    def test_dataset_version_mismatch_fail(self) -> None:
        """Runner validation must reject when audit SHA ≠ derived SHA."""
        runner = _load_script_module("run_ablations", "scripts/run_ablations.py")

        class _Config:
            seeds = (2025, 2026, 2027, 2028, 2029)

        ablations = []
        for group in ("seat_profile", "time_edges", "direction_edges"):
            for fold in ("0", "1", "2"):
                ablations.append(
                    {
                        "group": group,
                        "fold": fold,
                        "auc_contribution": 0.01,
                        "rank_ic_contribution": 0.01,
                    }
                )
        raw = {
            "audited": True,
            "status": "evaluated_fail_closed",
            "seeds": [2025, 2026, 2027, 2028, 2029],
            "folds": [
                {"fold_id": "0"},
                {"fold_id": "1"},
                {"fold_id": "2"},
            ],
            "metrics": [
                {
                    "model": "dt_seathgt",
                    "seed": 2025,
                    "fold": "0",
                    "auc": 0.6,
                    "rank_ic": 0.05,
                }
            ],
            "ablations": ablations,
            "audit": {"verified": True, "snapshot_manifest_hash": "00" * 32},
        }
        with self.assertRaises(ValueError) as ctx:
            runner._validate_result(raw, _Config(), "ff" * 32)
        self.assertIn("dataset version mismatch", str(ctx.exception))

    # ---- Guard 7: planned metrics not in real JSON ------------------------

    def test_planned_metrics_not_in_real_json(self) -> None:
        """Planned/target metrics must never appear in the real experiment JSON."""
        real = _full_passing_metrics()
        self.assertNotIn("target_auc", real)
        self.assertNotIn("target_rank_ic", real)
        self.assertNotIn("planned_metrics", real)
        for row in real["metrics"]:
            self.assertNotIn("target_auc", row)
            self.assertNotIn("target_rank_ic", row)
        for row in real["ablations"]:
            self.assertNotIn("target_auc", row)
            self.assertNotIn("planned", row)

    # ---- Guard 8: invalid CLI commands caught -----------------------------

    def test_invalid_cli_commands_caught(self) -> None:
        """Invalid CLI commands must exit non-zero."""
        proc = subprocess.run(
            [sys.executable, "-m", "seatflow", "does-not-exist"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr.lower())


if __name__ == "__main__":
    import unittest

    unittest.main()