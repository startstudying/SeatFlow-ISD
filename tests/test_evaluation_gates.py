import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.evaluation.gates import evaluate_promotion_decision, save_promotion_decision


def _passing_metrics() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in (11, 13, 17, 19, 23):
        for fold_index in range(3):
            fold = f"roll-{fold_index}"
            rows.append(
                {
                    "model": "strong_baseline",
                    "seed": seed,
                    "fold": fold,
                    "auc": 0.60,
                    "rank_ic": 0.050,
                    "topk_excess_return_after_cost": 0.0,
                    "max_drawdown": 0.12,
                }
            )
            rows.append(
                {
                    "model": "dt",
                    "seed": seed,
                    "fold": fold,
                    "auc": 0.615,
                    "rank_ic": 0.070,
                    "topk_excess_return_after_cost": 0.01,
                    "max_drawdown": 0.12,
                }
            )
    return {
        "audited": True,
        "candidate_model": "dt",
        "strong_baseline": "strong_baseline",
        "final_blind_manifest": {
            "split_status": "future_blind",
            "verified": True,
            "sealed": True,
            "independent": True,
        },
        "metrics": rows,
        "ablations": [
            {
                "group": group,
                "fold": f"roll-{index}",
                "auc_contribution": 0.002,
                "rank_ic_contribution": 0.003,
            }
            for group in ("seat_profile", "time_edges", "direction_edges")
            for index in range(3)
        ],
    }


class EvaluationGatesTest(unittest.TestCase):
    def test_passing_audited_multi_seed_multi_fold_decision_promotes(self) -> None:
        decision = evaluate_promotion_decision(_passing_metrics())

        self.assertTrue(decision.promoted)
        self.assertTrue(decision.model_valid)
        self.assertTrue(decision.backtest_valid)
        self.assertFalse(decision.competition_ready)
        self.assertEqual(decision.blocked_reasons, ())
        self.assertIn(
            "missing_verified_release_contract",
            decision.competition_blocked_reasons,
        )

    def test_competition_ready_requires_verified_release_contract(self) -> None:
        raw = _passing_metrics()
        raw["release_contract"] = {"verified": True, "contract_sha256": "d" * 64}

        decision = evaluate_promotion_decision(raw)

        self.assertTrue(decision.promoted)
        self.assertTrue(decision.competition_ready)
        self.assertTrue(decision.release_contract_verified)
        self.assertEqual(decision.competition_blocked_reasons, ())

    def test_missing_or_failed_inputs_save_negative_result_and_never_promote(self) -> None:
        raw = _passing_metrics()
        raw["audited"] = False
        raw["metrics"] = [
            row for row in raw["metrics"] if row["seed"] == 11 and row["fold"] == "roll-0"
        ]
        raw["ablations"] = []
        raw["final_blind_manifest"] = {
            "split_status": "contaminated_oos",
            "verified": True,
            "sealed": True,
            "independent": True,
        }

        decision = evaluate_promotion_decision(raw)

        self.assertFalse(decision.promoted)
        self.assertFalse(decision.model_valid)
        self.assertFalse(decision.backtest_valid)
        self.assertFalse(decision.competition_ready)
        self.assertTrue(decision.negative_result)
        self.assertIn("raw_metrics_not_audited", decision.blocked_reasons)
        self.assertIn("contaminated_oos_cannot_promote", decision.blocked_reasons)
        self.assertIn("single_or_missing_seed", decision.blocked_reasons)

        with TemporaryDirectory() as temporary_directory:
            output = save_promotion_decision(
                Path(temporary_directory) / "promotion.json", decision
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(payload["promoted"])
        self.assertTrue(payload["negative_result"])

    def test_threshold_miss_blocks_all_validity_claims(self) -> None:
        raw = _passing_metrics()
        for row in raw["metrics"]:
            if row["model"] == "dt":
                row["auc"] = 0.605

        decision = evaluate_promotion_decision(raw)

        self.assertFalse(decision.promoted)
        self.assertIn("auc_lift_below_threshold", decision.blocked_reasons)
        self.assertFalse(decision.model_valid)

    def test_two_seed_metrics_never_promote_by_default(self) -> None:
        raw = _passing_metrics()
        raw["metrics"] = [
            row for row in raw["metrics"] if row["seed"] in {11, 13}
        ]

        decision = evaluate_promotion_decision(raw)

        self.assertFalse(decision.promoted)
        self.assertIn("single_or_missing_seed", decision.blocked_reasons)

    def test_incomplete_candidate_baseline_seed_fold_pairs_fail_closed(self) -> None:
        raw = _passing_metrics()
        raw["metrics"] = [
            row
            for row in raw["metrics"]
            if not (
                row["model"] == "strong_baseline"
                and row["seed"] == 23
                and row["fold"] == "roll-2"
            )
        ]

        decision = evaluate_promotion_decision(raw)

        self.assertFalse(decision.promoted)
        self.assertIn("incomplete_seed_fold_metric_pairs", decision.blocked_reasons)


if __name__ == "__main__":
    unittest.main()
