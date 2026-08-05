"""Tests for the dataset seal (Task 13).

The :class:`Seal` is the final fail-closed guard on a built dataset.
Each :class:`HardGate` corresponds to one class of leak or integrity
violation. ``seal_dataset`` returns a :class:`SealReport` whose
``passed`` is ``True`` *only* if every gate passes — a single failed
gate makes the seal rejected.

Hard gates
----------

- ``PIT_AVAILABLE_TIME_VIOLATION`` — a feature's ``available_time`` is
  earlier than the sample's ``cutoff_time``.
- ``ID_LEAKAGE`` — two distinct stock_ids or seat_ids in the dataset
  share an identity alias that was not canonical at the time.
- ``IMMATURE_LABEL`` — a label's ``evaluation_date`` is before the
  sample's prediction_date (the label is not yet realized).
- ``CYCLE_CROSS_SPLIT`` — a cycle appears in more than one split.
- ``LABEL_PATH_INCONSISTENCY`` — a label's daily path is internally
  inconsistent (e.g. compound ≠ exit_price / entry_price - 1).
- ``PRIMARY_KEY_CONFLICT`` — two samples share the same ``sample_id``.
- ``DANGLING_GRAPH_REFERENCE`` — a sample references a node that does
  not exist in the graph snapshot.
"""

from __future__ import annotations

from datetime import datetime


from seatflow.dataset.manifest import DatasetManifest
from seatflow.dataset.samples import PoolType, PredictionSample
from seatflow.dataset.seal import HardGate, SealReport, seal_dataset


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _sample(
    sample_id: str = "s-001",
    cycle_id: str = "cyc-001",
    stock_id: str = "000001.SZ",
    *,
    prediction_date: str = "2026-01-05T00:00:00+00:00",
    cutoff_time: str = "2026-01-05T00:00:00+00:00",
    available_time: str = "2026-01-05T00:00:00+00:00",
    is_tradable: bool = True,
    is_backtest_eligible: bool = True,
    pool_type: PoolType = PoolType.DIRECT,
) -> PredictionSample:
    return PredictionSample(
        sample_id=sample_id,
        cycle_id=cycle_id,
        stock_id=stock_id,
        prediction_date=_utc(prediction_date),
        cutoff_time=_utc(cutoff_time),
        days_since_last_event=0,
        pool_type=pool_type,
        tradeability=__import__("seatflow.labels.tradeability", fromlist=["TradeabilityState"]).TradeabilityState.TRADABLE,
        available_time=_utc(available_time),
        is_tradable=is_tradable,
        is_backtest_eligible=is_backtest_eligible,
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_version="seatflow.v3.0",
        build_timestamp=_utc("2026-08-06T00:00:00+00:00"),
        source_manifest_hash="a" * 64,
        identity_policy_hash="b" * 64,
        cycle_policy_hash="c" * 64,
        feature_schema_hash="d" * 64,
        label_policy_hash="e" * 64,
        split_definition_hash="f" * 64,
        graph_schema_hash="0" * 64,
        row_counts={"samples": 1},
        node_counts={"stock": 1, "seat": 0},
        edge_counts={"participates": 0},
        quality_gate_results={},
        artifact_uris=[],
    )


# ── Clean pass ────────────────────────────────────────────────────────


class TestSealCleanPass:
    def test_clean_samples_pass_seal(self):
        samples = [_sample()]
        gate_results = {
            HardGate.PIT_AVAILABLE_TIME_VIOLATION: True,
            HardGate.ID_LEAKAGE: True,
            HardGate.IMMATURE_LABEL: True,
            HardGate.CYCLE_CROSS_SPLIT: True,
            HardGate.LABEL_PATH_INCONSISTENCY: True,
            HardGate.PRIMARY_KEY_CONFLICT: True,
            HardGate.DANGLING_GRAPH_REFERENCE: True,
        }
        report = seal_dataset(samples, _manifest(), gate_results)
        assert isinstance(report, SealReport)
        assert report.passed is True
        assert report.violation_count == 0
        assert report.violation_samples == []
        for gate in HardGate:
            assert report.gate_results[gate.value] is True


# ── Fail-closed behavior ──────────────────────────────────────────────


class TestSealFailClosed:
    def test_pit_violation_blocks_seal(self):
        samples = [_sample()]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.PIT_AVAILABLE_TIME_VIOLATION] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False
        assert report.violation_count >= 1
        assert any(
            s.sample_id == "s-001" for s in report.violation_samples
        )

    def test_id_leakage_blocks_seal(self):
        samples = [_sample()]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.ID_LEAKAGE] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False
        assert report.violation_count >= 1

    def test_immature_label_blocks_seal(self):
        samples = [_sample()]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.IMMATURE_LABEL] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False

    def test_cycle_cross_split_blocks_seal(self):
        samples = [_sample()]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.CYCLE_CROSS_SPLIT] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False

    def test_label_inconsistency_blocks_seal(self):
        samples = [_sample()]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.LABEL_PATH_INCONSISTENCY] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False


# ── Multi-violation accounting ────────────────────────────────────────


class TestSealViolationAccounting:
    def test_multiple_violations_counted(self):
        samples = [_sample(), _sample(sample_id="s-002", cycle_id="cyc-002")]
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.PIT_AVAILABLE_TIME_VIOLATION] = False
        gate_results[HardGate.ID_LEAKAGE] = False
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False
        assert report.violation_count == 2
        ids = {s.sample_id for s in report.violation_samples}
        assert ids == {"s-001", "s-002"}

    def test_missing_gate_results_block_seal(self):
        """A gate not in the result dict is treated as a failure
        (fail-closed)."""
        samples = [_sample()]
        gate_results = {HardGate.PIT_AVAILABLE_TIME_VIOLATION: True}  # others missing
        report = seal_dataset(samples, _manifest(), gate_results)
        assert report.passed is False
        for gate in HardGate:
            if gate not in gate_results:
                assert report.gate_results[gate.value] is False

    def test_violation_samples_carry_sample_object(self):
        sample = _sample()
        gate_results = {gate: True for gate in HardGate}
        gate_results[HardGate.PRIMARY_KEY_CONFLICT] = False
        report = seal_dataset([sample], _manifest(), gate_results)
        assert any(s is sample for s in report.violation_samples)
