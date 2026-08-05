"""Dataset seal — fail-closed integrity gate (Task 13).

The :class:`Seal` is the final fail-closed guard on a built dataset.
Each :class:`HardGate` corresponds to one class of leak or integrity
violation. :func:`seal_dataset` returns a :class:`SealReport` whose
``passed`` is ``True`` *only* if every gate passes — a single failed
gate (or a gate whose result is missing) makes the seal rejected.

Hard gates
----------

- ``PIT_AVAILABLE_TIME_VIOLATION`` — a feature's ``available_time`` is
  earlier than the sample's ``cutoff_time``.
- ``ID_LEAKAGE`` — two distinct stock_ids or seat_ids share an identity
  alias that was not canonical at the time.
- ``IMMATURE_LABEL`` — a label's ``evaluation_date`` is before the
  sample's prediction_date (the label is not yet realized).
- ``CYCLE_CROSS_SPLIT`` — a cycle appears in more than one split.
- ``LABEL_PATH_INCONSISTENCY`` — a label's daily path is internally
  inconsistent (e.g. compound ≠ exit_price / entry_price - 1).
- ``PRIMARY_KEY_CONFLICT`` — two samples share the same ``sample_id``.
- ``DANGLING_GRAPH_REFERENCE`` — a sample references a graph node that
  does not exist in the graph snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from seatflow.dataset.manifest import DatasetManifest
from seatflow.dataset.samples import PredictionSample


class HardGate(str, Enum):
    """A single class of integrity violation checked by the seal."""

    PIT_AVAILABLE_TIME_VIOLATION = "PIT_AVAILABLE_TIME_VIOLATION"
    ID_LEAKAGE = "ID_LEAKAGE"
    IMMATURE_LABEL = "IMMATURE_LABEL"
    CYCLE_CROSS_SPLIT = "CYCLE_CROSS_SPLIT"
    LABEL_PATH_INCONSISTENCY = "LABEL_PATH_INCONSISTENCY"
    PRIMARY_KEY_CONFLICT = "PRIMARY_KEY_CONFLICT"
    DANGLING_GRAPH_REFERENCE = "DANGLING_GRAPH_REFERENCE"


@dataclass(frozen=True)
class SealReport:
    """Result of sealing a dataset.

    Attributes
    ----------
    passed:
        ``True`` iff every gate passed (no violations). A single failed
        gate — or a gate whose result is missing — makes this ``False``.
    gate_results:
        Mapping of gate name → bool (``True`` = passed). Gates missing
        from the input are recorded as ``False`` (fail-closed).
    violation_count:
        Number of gates that failed.
    violation_samples:
        The samples implicated by a failed gate. For a gate-level failure
        (no specific sample), all input samples are attached so callers
        can inspect them.
    """

    passed: bool
    gate_results: dict[str, bool]
    violation_count: int
    violation_samples: list[PredictionSample]


def seal_dataset(
    samples: Sequence[PredictionSample],
    manifest: DatasetManifest,
    gate_results: dict[HardGate, bool],
) -> SealReport:
    """Run the fail-closed seal on a built dataset.

    Parameters
    ----------
    samples:
        The prediction samples produced by :func:`build_samples`.
    manifest:
        The dataset manifest produced alongside the samples.
    gate_results:
        Mapping of :class:`HardGate` → pass/fail. Every gate must be
        present and ``True`` for the seal to pass. A gate that is missing
        from this mapping is treated as a failure (fail-closed).

    Returns
    -------
    SealReport
        The sealed report. ``passed`` is ``True`` only if every gate
        passed.
    """
    # Build a complete gate-results dict, defaulting missing gates to
    # False (fail-closed).
    complete: dict[str, bool] = {}
    failed_gates: list[HardGate] = []
    for gate in HardGate:
        passed = gate_results.get(gate, False)
        complete[gate.value] = passed
        if not passed:
            failed_gates.append(gate)

    violation_count = len(failed_gates)

    # Attach samples implicated by failed gates. For a gate-level failure
    # (no specific sample identified), attach all input samples so callers
    # can inspect them.
    if failed_gates and samples:
        violation_samples = list(samples)
    else:
        violation_samples = []

    passed = violation_count == 0

    return SealReport(
        passed=passed,
        gate_results=complete,
        violation_count=violation_count,
        violation_samples=violation_samples,
    )
