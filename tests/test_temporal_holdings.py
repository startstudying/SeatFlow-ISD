from __future__ import annotations

import unittest
from typing import cast

from seatflow.evaluation.temporal_holdings import (
    analyze_temporal_holdings,
    cross_seed_same_day_jaccard,
)


class TemporalHoldingsTest(unittest.TestCase):
    def test_temporal_analysis_reports_turnover_and_exit_reasons(self) -> None:
        rows = [
            # day 1
            {"prediction_date": "2026-01-01", "stock_id": "A", "score": 10},
            {"prediction_date": "2026-01-01", "stock_id": "B", "score": 9},
            {"prediction_date": "2026-01-01", "stock_id": "C", "score": 8},
            {"prediction_date": "2026-01-01", "stock_id": "D", "score": 7},
            {"prediction_date": "2026-01-01", "stock_id": "E", "score": 6},
            {"prediction_date": "2026-01-01", "stock_id": "F", "score": 5},
            # day 2
            {"prediction_date": "2026-01-02", "stock_id": "A", "score": 11},
            {"prediction_date": "2026-01-02", "stock_id": "C", "score": 8},
            {"prediction_date": "2026-01-02", "stock_id": "D", "score": 7},
            {"prediction_date": "2026-01-02", "stock_id": "E", "score": 6},
            {"prediction_date": "2026-01-02", "stock_id": "F", "score": 5},
            # day 3
            {"prediction_date": "2026-01-03", "stock_id": "A", "score": 11},
            {"prediction_date": "2026-01-03", "stock_id": "C", "score": 9},
            {"prediction_date": "2026-01-03", "stock_id": "D", "score": 8},
            {"prediction_date": "2026-01-03", "stock_id": "E", "score": 4},
            {"prediction_date": "2026-01-03", "stock_id": "G", "score": 3},
            # day 4
            {"prediction_date": "2026-01-04", "stock_id": "X", "score": 20},
            {"prediction_date": "2026-01-04", "stock_id": "Y", "score": 19},
            {"prediction_date": "2026-01-04", "stock_id": "Z", "score": 18},
            {"prediction_date": "2026-01-04", "stock_id": "W", "score": 17},
            {"prediction_date": "2026-01-04", "stock_id": "V", "score": 16},
        ]
        report = analyze_temporal_holdings(rows)
        summaries = cast(dict[str, dict[str, float]], report["summaries"])
        exit_reasons = cast(dict[str, float], report["exit_reasons"])
        daily = cast(list[dict[str, object]], report["daily"])

        top5 = summaries["5"]
        self.assertAlmostEqual(top5["full_turnover_ratio"], 1.0 / 4.0)
        self.assertAlmostEqual(top5["zero_turnover_ratio"], 0.0)
        self.assertGreater(top5["one_way_turnover"], 0.0)

        self.assertEqual(
            exit_reasons["candidate_exit"]
            + exit_reasons["rank_change"]
            + exit_reasons["other_unavailable"],
            exit_reasons["total"],
        )
        self.assertGreater(exit_reasons["candidate_exit"], 0.0)

        # day2 to day3 should retain 4, replace 1 for top5
        day3 = next(
            row
            for row in daily
            if row["prediction_date"] == "2026-01-03" and row["top_k"] == 5
        )
        self.assertEqual(day3["retained"], 4)
        self.assertEqual(day3["replaced"], 1)

    def test_cross_seed_same_day_jaccard(self) -> None:
        rows = [
            {"prediction_date": "2026-01-01", "seed": 1, "stock_id": "A", "score": 3},
            {"prediction_date": "2026-01-01", "seed": 1, "stock_id": "B", "score": 2},
            {"prediction_date": "2026-01-01", "seed": 1, "stock_id": "C", "score": 1},
            {"prediction_date": "2026-01-01", "seed": 2, "stock_id": "A", "score": 3},
            {"prediction_date": "2026-01-01", "seed": 2, "stock_id": "C", "score": 2},
            {"prediction_date": "2026-01-01", "seed": 2, "stock_id": "B", "score": 1},
            {"prediction_date": "2026-01-01", "seed": 3, "stock_id": "B", "score": 3},
            {"prediction_date": "2026-01-01", "seed": 3, "stock_id": "A", "score": 2},
            {"prediction_date": "2026-01-01", "seed": 3, "stock_id": "C", "score": 1},
        ]

        result = cross_seed_same_day_jaccard(rows, top_k=2)
        self.assertAlmostEqual(result["mean_jaccard"], 5.0 / 9.0)
        self.assertEqual(result["pair_count"], 3.0)


if __name__ == "__main__":
    unittest.main()
