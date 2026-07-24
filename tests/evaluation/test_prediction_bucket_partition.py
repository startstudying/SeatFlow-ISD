from __future__ import annotations

from seatflow.evaluation.topk_diagnostics import partition_prediction_buckets


def test_prediction_buckets_are_disjoint_and_exhaustive() -> None:
    rows = [
        {"stock_id": f"{index:06d}.SZ", "score": float(index % 3)}
        for index in range(23)
    ]

    buckets = partition_prediction_buckets(rows)
    flattened = [row for bucket in buckets for row in bucket]

    assert len(buckets) == 10
    assert len(flattened) == len(rows)
    assert {id(row) for row in flattened} == {id(row) for row in rows}
    assert max(map(len, buckets)) - min(map(len, buckets)) <= 1
