import pytest

from seatflow.evaluation.r3_6_audits import _mapping_sequence


def test_mapping_sequence_rejects_non_row_payload() -> None:
    with pytest.raises(TypeError, match="list of mappings"):
        _mapping_sequence(["not-a-row"])
