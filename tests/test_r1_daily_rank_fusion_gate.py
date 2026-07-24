from __future__ import annotations

import numpy as np
import pytest

from scripts.run_r1_daily_rank_fusion_gate import _daily_rank_fusion


def test_daily_rank_fusion_averages_date_local_member_ranks() -> None:
    scores = np.asarray(
        [
            [0.1, 0.8],
            [0.2, 0.4],
            [0.3, 0.2],
            [0.9, 0.1],
            [0.5, 0.3],
        ]
    )
    dates = np.asarray(["2026-01-01"] * 3 + ["2026-01-02"] * 2)

    fused = _daily_rank_fusion(scores, dates)

    np.testing.assert_allclose(fused[:3], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(fused[3:], [0.5, 0.5])


def test_daily_rank_fusion_rejects_unaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned dates"):
        _daily_rank_fusion(np.ones((3, 2)), np.asarray(["2026-01-01"]))
