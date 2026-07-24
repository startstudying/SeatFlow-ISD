from __future__ import annotations

from seatflow.evaluation.topk_diagnostics import topk_diagnostics


def test_topk_curve_reproducible_under_input_reordering() -> None:
    rows = [
        {
            "prediction_date": day,
            "stock_id": f"{index:06d}.SZ",
            "score": float(index % 4),
            "realized_return": float(index - 10) / 100,
        }
        for day in ("2026-01-02", "2026-01-03")
        for index in range(20)
    ]

    forward = topk_diagnostics(rows)
    reversed_result = topk_diagnostics(list(reversed(rows)))

    assert forward == reversed_result
    assert set(forward["topk_curve"]) == {
        "top_1",
        "top_3",
        "top_5",
        "top_10",
        "top_20",
        "top_10pct",
        "top_20pct",
    }
