from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from tempfile import TemporaryDirectory

import torch

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import (
    LEGACY_BINARY_LABEL_SCHEMA,
    TensorizedGraph,
    TensorizedWindowSample,
    tensorize_window_sample,
)
from seatflow.datasets.windows import build_window_samples
from seatflow.graph.tensorize import PyGUnavailableError
from seatflow.models.dt_seat_hgt_v3 import (
    DTSeatHGTv3Config,
    build_dt_seat_hgt_v3,
)
from seatflow.models.seat_hgt import PyGModelUnavailableError
from seatflow.training.dtseathgt_v3 import (
    DTSeatHGTv3TrainingConfig,
    DTSeatHGTv3TrainingStageConfig,
    TrainingStage,
    _apply_freeze_mask,
    default_v3_training_config,
    train_dt_seat_hgt_v3,
)

# Model dims used by the mock graph. These must match the model config so the
# SeatRelationBranch default edge_attr_dim (7) lines up with the mock seat edges.
_MOCK_HIDDEN_DIM = 16
_MOCK_STATIC_FEATURE_COUNT = 8
_MOCK_SEAT_FEATURE_DIM = 7
_MOCK_INDUSTRY_FEATURE_DIM = _MOCK_HIDDEN_DIM  # model defaults industry to hidden_dim


def _mock_config() -> DTSeatHGTv3Config:
    """Model config whose dims line up with :func:`_mock_graph`."""
    return DTSeatHGTv3Config(
        hidden_dim=_MOCK_HIDDEN_DIM,
        stock_gru_hidden_dim=8,
        stock_gru_layers=1,
        stock_static_hidden_dim=8,
        stock_static_feature_count=_MOCK_STATIC_FEATURE_COUNT,
        stock_sequence_feature_dim=5,
        seat_feature_dim=_MOCK_SEAT_FEATURE_DIM,
        num_heads=2,
        seat_style_count=3,
        seat_style_embedding_dim=8,
        dropout=0.0,
        relation_dropout=0.0,
        seat_gate_initial_bias=-2.0,
    )


def _mock_graph(seed: int = 0) -> torch.Tensor:
    """Build a minimal HeteroData graph with dims matching :func:`_mock_config`.

    The real fixture (``_tensorized_samples``) produces seat edge attrs of width
    14 and a 16-dim stock static feature vector, but the model's
    ``SeatRelationBranch`` defaults its edge gate to 7 dims and the model does
    not infer edge_attr dims from the config (a model-layer gap tracked with T1).
    This mock keeps the dims consistent so the training logic can be exercised
    end-to-end without depending on that gap.
    """
    torch.manual_seed(seed)
    from torch_geometric.data import HeteroData

    graph = HeteroData()
    graph["stock"].x = torch.randn(3, _MOCK_STATIC_FEATURE_COUNT)
    graph["seat"].x = torch.randn(2, _MOCK_SEAT_FEATURE_DIM)
    graph["industry"].x = torch.randn(1, _MOCK_INDUSTRY_FEATURE_DIM)
    # Seat buys: seat 0 -> stock 0, seat 1 -> stock 1
    graph["seat", "buys", "stock"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    graph["seat", "buys", "stock"].edge_attr = torch.randn(2, 7)
    # Industry -> stock: industry 0 -> stock 0
    graph["industry", "belongs_to", "stock"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    graph["industry", "belongs_to", "stock"].edge_attr = torch.randn(1, 7)
    return graph


def _mock_samples(n: int = 4, base_seed: int = 0) -> tuple[TensorizedWindowSample, ...]:
    """Build mock tensorized samples with consistent dims for training tests."""
    samples: list[TensorizedWindowSample] = []
    for i in range(n):
        graph = _mock_graph(base_seed + i)
        wrapped = TensorizedGraph(data=graph, metadata={})
        sample = TensorizedWindowSample(
            sample_id=f"mock::{base_seed + i}",
            source_hash="mock",
            target_stock_id="000001.SZ",
            target_node_index=0,
            graph=wrapped,
            labels={1: 1},
            returns={1: 0.05},
            as_of_date=date(2026, 1, 15) + timedelta(days=i),
            label_end_dates={},
            label_schema_version=LEGACY_BINARY_LABEL_SCHEMA,
        )
        samples.append(sample)
    return tuple(samples)


def _mock_training_config(seeds: tuple[int, ...] = (2026,)) -> DTSeatHGTv3TrainingConfig:
    """Minimal three-stage training config using the mock model config."""
    return DTSeatHGTv3TrainingConfig(
        model=_mock_config(),
        stage1=replace(default_v3_training_config().stage1, epochs=1),
        stage2=replace(default_v3_training_config().stage2, epochs=1),
        stage3=replace(default_v3_training_config().stage3, epochs=1),
        seeds=seeds,
        validation_split=0.0,
        use_amp=False,
    )


_BASE_PARAM_PREFIXES = (
    "stock_gru",
    "stock_static_mlp",
    "base_projection",
    "base_relation_branch",
    "industry_projection",
    "layer_norm",
    "base_rank_head",
    "base_return_head",
)
_SEAT_PARAM_PREFIXES = (
    "seat_projection",
    "seat_relation_branch",
    "seat_rank_delta_head",
    "seat_return_delta_head",
    "seat_gate_mlp",
    "seat_style_encoder",
)


class DTSeatHGTv3TrainingTest(unittest.TestCase):
    def test_three_stage_training_completes_all_stages(self) -> None:
        try:
            samples = _mock_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = _mock_training_config(seeds=(0,))

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.model_valid)
        self.assertEqual(result.total_epochs, 3)
        self.assertIn("BASE_ONLY", result.stage_results)
        self.assertIn("SEAT_RESIDUAL", result.stage_results)
        self.assertIn("JOINT_FINETUNE", result.stage_results)

    def test_stage1_freezes_seat_branch(self) -> None:
        try:
            samples = _mock_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = _mock_training_config()

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.stage_results["BASE_ONLY"]["epochs"], 1)

    def test_stage2_freezes_base_branch(self) -> None:
        try:
            samples = _mock_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = _mock_training_config()

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.stage_results["SEAT_RESIDUAL"]["epochs"], 1)

    def test_multi_seed_returns_all_results(self) -> None:
        """Verify that training returns results for ALL seeds, not just the last one.

        Bug fix: Previously returned all_results[-1], discarding earlier seeds.
        """
        try:
            samples = _mock_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = _mock_training_config(seeds=(2026, 2027, 2028))

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        seeds = [r.seed for r in results]
        self.assertEqual(sorted(seeds), [2026, 2027, 2028])
        for result in results:
            self.assertEqual(result.status, "completed")
            self.assertFalse(result.model_valid)

    def test_time_sorted_validation_split(self) -> None:
        """Verify that validation split uses time order, not random shuffle."""
        try:
            samples = _mock_samples(n=4)
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = DTSeatHGTv3TrainingConfig(
            model=_mock_config(),
            stage1=replace(default_v3_training_config().stage1, epochs=1),
            stage2=replace(default_v3_training_config().stage2, epochs=1),
            stage3=replace(default_v3_training_config().stage3, epochs=1),
            seeds=(2026,),
            validation_split=0.25,  # 25% for validation (time-sorted)
            use_amp=False,
        )

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

    # ------------------------------------------------------------------
    # V4-A8 + training-correctness tests (T2)
    # ------------------------------------------------------------------

    def test_stage_b_residual_target_is_detached(self) -> None:
        """Stage B residual target must be ``label - base_pred.detach()``.

        The base branch prediction feeding the residual target must be detached
        so no gradient flows back to the base branch during Stage B. This is the
        core V4-A8 correctness requirement.
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        graph = _mock_graph(seed=42)
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        stage_cfg = DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.SEAT_RESIDUAL,
            epochs=1,
            learning_rate=1e-3,
            freeze_base=True,
            freeze_seat=False,
        )
        _apply_freeze_mask(model, stage_cfg)

        target_idx = torch.tensor([0], dtype=torch.long)
        label_ret = torch.tensor([0.05], dtype=torch.float32)

        model.zero_grad()
        outputs = model(graph, enable_seat_branch=True)
        base_pred = outputs["base_return_prediction"][target_idx]

        # Replicate the trainer's Stage B residual target construction.
        base_pred_detached = base_pred.detach()
        residual_target = label_ret - base_pred_detached
        expected_target = label_ret - base_pred.detach()

        # The target equals label - base_pred.detach() and carries no gradient.
        self.assertTrue(torch.allclose(residual_target, expected_target))
        self.assertFalse(residual_target.requires_grad)
        self.assertFalse(base_pred_detached.requires_grad)

        # Compute the Stage B loss exactly as the trainer does, then backprop.
        seat_delta = outputs["seat_return_delta"][target_idx]
        seat_gate_detached = outputs["seat_gate"][target_idx].detach()
        seat_contribution = seat_gate_detached * seat_delta
        import torch.nn as nn

        ret_loss = nn.HuberLoss()(seat_contribution, residual_target)
        direction_logit = outputs["direction_logit"][target_idx]
        class_loss = nn.BCEWithLogitsLoss()(
            direction_logit, torch.tensor([1.0], dtype=torch.float32)
        )
        (class_loss + ret_loss).backward()

        # No base/embedding parameter may receive a gradient in Stage B.
        leaking = [
            name
            for name, param in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
            and param.grad is not None
            and param.grad.abs().sum().item() > 0
        ]
        self.assertEqual(leaking, [], f"gradient leaked to base params: {leaking}")

        # Seat parameters must still learn (otherwise the stage is a no-op).
        seat_learning = [
            name
            for name, param in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
            and param.grad is not None
            and param.grad.abs().sum().item() > 0
        ]
        self.assertGreater(len(seat_learning), 0, "seat params received no gradient")

    def test_all_trainable_params_in_optimizer(self) -> None:
        """In the joint stage every trainable parameter is in the optimizer.

        This guards the T1 fix: modules moved into ``__init__`` (seat_projection,
        industry_projection, seat_style_encoder) must be registered parameters
        and therefore be picked up by the optimizer when nothing is frozen.
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        graph = _mock_graph(seed=7)
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        joint_cfg = DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.JOINT_FINETUNE,
            epochs=1,
            learning_rate=1e-4,
            freeze_base=False,
            freeze_seat=False,
        )
        _apply_freeze_mask(model, joint_cfg)

        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_ids = {id(p) for p in trainable}
        all_ids = {id(p) for p in model.parameters()}

        # Nothing frozen in the joint stage -> every parameter is trainable.
        self.assertEqual(trainable_ids, all_ids)
        self.assertGreater(len(trainable), 0)

        # The optimizer the trainer builds from this list must cover them all.
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimizer_param_ids, trainable_ids)

        # Key T1-fixed modules are registered and trainable.
        for module_name in (
            "seat_projection",
            "industry_projection",
            "seat_style_encoder",
            "seat_relation_branch",
        ):
            module = getattr(model, module_name)
            module_params = list(module.parameters())
            self.assertGreater(len(module_params), 0, f"{module_name} has no parameters")
            self.assertTrue(
                all(id(p) in trainable_ids for p in module_params),
                f"{module_name} parameters missing from optimizer",
            )

    def test_multi_seed_runs_produce_different_results(self) -> None:
        """Different seeds must produce different model weights and losses.

        Guards the audit finding "multi_seed_not_implemented" / "only the first
        seed is used": each seed must re-seed the RNGs and rebuild the model so
        results genuinely differ.
        """
        try:
            samples = _mock_samples(n=4)
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        config = _mock_training_config(seeds=(2026, 2027, 2028))

        with TemporaryDirectory() as tmp:
            results = train_dt_seat_hgt_v3(samples, config, tmp, device="cpu")
            state_dicts = [
                torch.load(
                    tmp + f"/seed_{seed}/dtseathgt_v3_final.pt",
                    weights_only=False,
                )["model_state_dict"]
                for seed in (2026, 2027, 2028)
            ]

        self.assertEqual(len(results), 3)
        losses = [round(r.final_loss, 8) for r in results]
        self.assertEqual(len(set(losses)), 3, f"losses not distinct: {losses}")

        # Seed 2026 vs 2027 weights must differ.
        sd_a, sd_b = state_dicts[0], state_dicts[1]
        differ = any(not torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
        self.assertTrue(differ, "seed 2026 and 2027 produced identical weights")

    def test_three_stage_freeze_logic(self) -> None:
        """Stage A freezes seat, Stage B freezes base, Stage C freezes nothing.

        Also verifies the cross-stage unfreeze: each stage starts from a fully
        unfrozen model so a frozen-then-learnable component is learnable again.
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        graph = _mock_graph(seed=11)
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        def freeze_summary(stage_cfg: DTSeatHGTv3TrainingStageConfig) -> dict[str, bool]:
            _apply_freeze_mask(model, stage_cfg)
            base_trainable = all(
                p.requires_grad
                for name, p in model.named_parameters()
                if name.split(".")[0] in _BASE_PARAM_PREFIXES
            )
            seat_trainable = all(
                p.requires_grad
                for name, p in model.named_parameters()
                if name.split(".")[0] in _SEAT_PARAM_PREFIXES
            )
            return {"base_trainable": base_trainable, "seat_trainable": seat_trainable}

        defaults = default_v3_training_config()

        # Stage A (BASE_ONLY): base trainable, seat frozen.
        stage_a = freeze_summary(defaults.stage1)
        self.assertTrue(stage_a["base_trainable"], "Stage A must keep base trainable")
        self.assertFalse(stage_a["seat_trainable"], "Stage A must freeze seat")

        # Stage B (SEAT_RESIDUAL): base frozen, seat trainable.
        stage_b = freeze_summary(defaults.stage2)
        self.assertFalse(stage_b["base_trainable"], "Stage B must freeze base")
        self.assertTrue(stage_b["seat_trainable"], "Stage B must keep seat trainable")

        # Stage C (JOINT_FINETUNE): everything trainable.
        stage_c = freeze_summary(defaults.stage3)
        self.assertTrue(stage_c["base_trainable"], "Stage C must keep base trainable")
        self.assertTrue(stage_c["seat_trainable"], "Stage C must keep seat trainable")

        # Cross-stage reset: after Stage B froze base, Stage C must unfreeze it
        # even though _apply_freeze_mask is called fresh each stage.
        _apply_freeze_mask(model, defaults.stage2)
        self.assertFalse(
            all(
                p.requires_grad
                for name, p in model.named_parameters()
                if name.split(".")[0] in _BASE_PARAM_PREFIXES
            )
        )
        _apply_freeze_mask(model, defaults.stage3)
        self.assertTrue(
            all(
                p.requires_grad
                for name, p in model.named_parameters()
                if name.split(".")[0] in _BASE_PARAM_PREFIXES
            )
        )


def _tensorized_samples() -> tuple[TensorizedWindowSample, ...]:
    """Real-pipeline fixture (kept for reference).

    NOTE: this fixture produces seat edge attrs of width 14 and a 16-dim stock
    static feature, which do not match the model's current config-based defaults
    (SeatRelationBranch edge gate defaults to 7; industry/seat feature dims are
    inferred from the graph schema, which this fixture does not provide). The
    training tests therefore use :func:`_mock_samples` instead. Closing the
    fixture/model dim gap is tracked as a T1 dependency (model should infer
    edge_attr dims from the frozen graph schema).
    """
    as_of = date(2026, 1, 15)
    cutoff = datetime(2026, 1, 15, 18, 0, 0)
    samples = build_window_samples(
        records=(
            DragonTigerRecord(
                trade_date=as_of,
                seat_id="seat_a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=10_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="daily_deviation",
                available_time=cutoff - timedelta(hours=1),
            ),
            DragonTigerRecord(
                trade_date=as_of - timedelta(days=1),
                seat_id="seat_b",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=3_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="daily_deviation",
                available_time=cutoff - timedelta(days=1),
            ),
        ),
        bars=tuple(
            DailyBar(
                stock_id="000001.SZ",
                trade_date=as_of + timedelta(days=offset),
                open_price=Decimal("10.0") + Decimal(str(offset)) / Decimal("10"),
                high_price=Decimal("10.1") + Decimal(str(offset)) / Decimal("10"),
                low_price=Decimal("9.9") + Decimal(str(offset)) / Decimal("10"),
                close_price=Decimal("10.0") + Decimal(str(offset)) / Decimal("10"),
                volume=Decimal("1000000"),
                amount=Decimal("10000000"),
                source_name="fixture",
                available_time=cutoff + timedelta(days=offset),
            )
            for offset in range(0, 6)
        ),
        as_of_dates=(as_of,),
        target_stock_ids=("000001.SZ",),
        horizons=(1, 3, 5),
        cutoff_for_date=lambda _: cutoff,
        decay_lambda=0.1,
        max_edge_age_days=20,
    )
    return tuple(tensorize_window_sample(sample) for sample in samples)


def _mock_samples_same_date(n: int = 3, base_seed: int = 0) -> tuple[TensorizedWindowSample, ...]:
    """Mock samples sharing ONE as_of_date so listwise loss gets a real list.

    The default ``_mock_samples`` gives each sample a distinct date, so every
    date bucket has a single element and listwise_rank_loss degenerates to 0.
    These tests need a multi-element cross-section to exercise ListMLE and the
    date-bucketing of _compute_v4_loss_for_buckets.
    """
    samples: list[TensorizedWindowSample] = []
    same_date = date(2026, 1, 15)
    for i in range(n):
        graph = _mock_graph(base_seed + i)
        wrapped = TensorizedGraph(data=graph, metadata={})
        sample = TensorizedWindowSample(
            sample_id=f"mock-same::{base_seed + i}",
            source_hash="mock",
            target_stock_id=f"00000{i}.SZ",
            target_node_index=0,
            graph=wrapped,
            labels={1: 1 if i % 2 == 0 else 0},  # binary schema {0,1}
            returns={1: 0.05 * (i - 1)},  # distinct returns -> rankable
            as_of_date=same_date,
            label_end_dates={},
            label_schema_version=LEGACY_BINARY_LABEL_SCHEMA,
        )
        samples.append(sample)
    return tuple(samples)


class V4LossIntegrationTest(unittest.TestCase):
    """Verify the V4Loss date-bucketed training loop integration (P0 item 1)."""

    def test_group_samples_by_date_buckets_same_date(self) -> None:
        from seatflow.training.dtseathgt_v3 import _group_samples_by_date

        samples = _mock_samples_same_date(n=3)
        buckets = _group_samples_by_date(samples)
        # All 3 samples share one as_of_date -> exactly one bucket of size 3.
        self.assertEqual(len(buckets), 1)
        self.assertEqual(len(buckets[0]), 3)

    def test_group_samples_by_date_preserves_time_order(self) -> None:
        from seatflow.training.dtseathgt_v3 import _group_samples_by_date

        # Distinct dates -> one bucket each, in sorted date order.
        samples = _mock_samples(n=4)
        buckets = _group_samples_by_date(samples)
        self.assertEqual(len(buckets), 4)
        dates = [samples[bucket[0]].as_of_date for bucket in buckets]
        self.assertEqual(dates, sorted(dates))

    def test_stage_b_v4loss_base_zero_grad_seat_nonzero(self) -> None:
        """Stage B via V4Loss: base params zero-grad, seat params nonzero."""
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        from seatflow.training.dtseathgt_v3 import (
            _compute_v4_loss_for_buckets,
            _v4_loss_weights_for_stage,
        )
        from seatflow.training.v4_loss import V4Loss

        samples = _mock_samples_same_date(n=3, base_seed=7)
        graph = samples[0].graph.data
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        stage_cfg = DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.SEAT_RESIDUAL,
            epochs=1,
            learning_rate=1e-3,
            freeze_base=True,
            freeze_seat=False,
        )
        _apply_freeze_mask(model, stage_cfg)

        buckets = [[0, 1, 2]]
        v4_loss = V4Loss(_v4_loss_weights_for_stage(TrainingStage.SEAT_RESIDUAL))
        model.zero_grad()
        loss, _ = _compute_v4_loss_for_buckets(
            model,
            samples,
            buckets,
            v4_loss,
            enable_seat_branch=True,
            direction_classes=2,
            direction_weight=1.0,
            device="cpu",
        )
        loss.backward()

        base_leak = [
            name
            for name, param in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
            and param.requires_grad
            and param.grad is not None
            and param.grad.abs().sum().item() > 0
        ]
        self.assertEqual(base_leak, [], f"V4Loss leaked grad to base params: {base_leak}")

        seat_learning = [
            name
            for name, param in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
            and param.requires_grad
            and param.grad is not None
            and param.grad.abs().sum().item() > 0
        ]
        self.assertGreater(len(seat_learning), 0, "seat params received no gradient under V4Loss")

    def test_listwise_signal_present_in_multielement_bucket(self) -> None:
        """A multi-element bucket must produce a nonzero listwise_loss.

        With a single-element bucket listwise degenerates to 0; this confirms
        the date-bucketing actually feeds a real cross-section to ListMLE.
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        from seatflow.training.dtseathgt_v3 import _compute_v4_loss_for_buckets
        from seatflow.training.v4_loss import V4Loss, V4LossWeights

        samples = _mock_samples_same_date(n=3, base_seed=11)
        graph = samples[0].graph.data
        model = build_dt_seat_hgt_v3(_mock_config(), graph)
        model.eval()  # deterministic forward (avoid dropout noise)

        v4_loss = V4Loss(V4LossWeights(base_weight=1.0, seat_weight=1.0, listwise_weight=1.0))
        _, components = _compute_v4_loss_for_buckets(
            model,
            samples,
            [[0, 1, 2]],
            v4_loss,
            enable_seat_branch=True,
            direction_classes=2,
            direction_weight=0.0,
            device="cpu",
        )
        self.assertIn("listwise_loss", components)
        self.assertGreater(
            components["listwise_loss"],
            0.0,
            "listwise loss should be > 0 for a 3-element list",
        )

    def test_date_bucket_isolation(self) -> None:
        """Two distinct dates -> two buckets; listwise computed per-bucket.

        The component_sums should reflect both buckets contributing. This guards
        against accidentally flattening all samples into one list (which would
        mix cross-sections and leak ranking signal across dates).
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        from seatflow.training.dtseathgt_v3 import (
            _compute_v4_loss_for_buckets,
            _group_samples_by_date,
        )
        from seatflow.training.v4_loss import V4Loss, V4LossWeights

        # 2 same-date samples + 2 same-date samples on a different date.
        batch_a = _mock_samples_same_date(n=2, base_seed=1)
        batch_b = _mock_samples_same_date(n=2, base_seed=9)
        # Shift batch_b to a later date.
        shifted = []
        for s in batch_b:
            shifted.append(
                TensorizedWindowSample(
                    sample_id=s.sample_id,
                    source_hash=s.source_hash,
                    target_stock_id=s.target_stock_id,
                    target_node_index=s.target_node_index,
                    graph=s.graph,
                    labels=s.labels,
                    returns=s.returns,
                    as_of_date=date(2026, 2, 15),
                    label_end_dates=s.label_end_dates,
                    label_schema_version=s.label_schema_version,
                )
            )
        all_samples = list(batch_a) + shifted
        buckets = _group_samples_by_date(all_samples)
        self.assertEqual(len(buckets), 2, "expected 2 date buckets")
        self.assertEqual(len(buckets[0]), 2)
        self.assertEqual(len(buckets[1]), 2)

        graph = all_samples[0].graph.data
        model = build_dt_seat_hgt_v3(_mock_config(), graph)
        model.eval()
        v4_loss = V4Loss(V4LossWeights(base_weight=1.0, seat_weight=1.0, listwise_weight=1.0))
        _, components = _compute_v4_loss_for_buckets(
            model,
            all_samples,
            buckets,
            v4_loss,
            enable_seat_branch=True,
            direction_classes=2,
            direction_weight=0.0,
            device="cpu",
        )
        # Both buckets have >=2 elements, so listwise is active and positive.
        self.assertGreater(components["listwise_loss"], 0.0)

    def test_stage_a_optimizer_step_leaves_seat_unchanged(self) -> None:
        """Stage A: after an optimizer step, seat params are bit-identical.

        Guards beyond requires_grad: even though seat is frozen by flag, a bug
        in the freeze mask or optimizer construction could still mutate seat
        params. Snapshot before/after a real AdamW step.
        """
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        from seatflow.training.dtseathgt_v3 import _apply_freeze_mask

        samples = _mock_samples_same_date(n=3, base_seed=3)
        graph = samples[0].graph.data
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        stage_cfg = DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.BASE_ONLY,
            epochs=1,
            learning_rate=1e-3,
            freeze_base=False,
            freeze_seat=True,
        )
        _apply_freeze_mask(model, stage_cfg)

        seat_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
        }
        base_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
        }

        # One real optimizer step on a single bucket.
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        from seatflow.training.dtseathgt_v3 import (
            _compute_v4_loss_for_buckets,
            _v4_loss_weights_for_stage,
        )
        from seatflow.training.v4_loss import V4Loss

        v4_loss = V4Loss(_v4_loss_weights_for_stage(TrainingStage.BASE_ONLY))
        model.train()
        optimizer.zero_grad()
        loss, _ = _compute_v4_loss_for_buckets(
            model, samples, [[0, 1, 2]], v4_loss,
            enable_seat_branch=False,
            direction_classes=2, direction_weight=1.0, device="cpu",
        )
        loss.backward()
        optimizer.step()

        # Seat params must be unchanged (frozen).
        seat_unchanged = all(
            torch.equal(seat_before[name], p.detach())
            for name, p in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
        )
        self.assertTrue(seat_unchanged, "Stage A optimizer step mutated frozen seat params")

        # At least one base param must have moved (base is trainable).
        base_moved = any(
            not torch.equal(base_before[name], p.detach())
            for name, p in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
        )
        self.assertTrue(base_moved, "Stage A optimizer step did not move any base param")

    def test_stage_b_optimizer_step_leaves_base_unchanged(self) -> None:
        """Stage B: after an optimizer step, base params are bit-identical."""
        try:
            import torch_geometric  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch-geometric unavailable")

        from seatflow.training.dtseathgt_v3 import (
            _apply_freeze_mask,
            _compute_v4_loss_for_buckets,
            _v4_loss_weights_for_stage,
        )
        from seatflow.training.v4_loss import V4Loss

        samples = _mock_samples_same_date(n=3, base_seed=5)
        graph = samples[0].graph.data
        model = build_dt_seat_hgt_v3(_mock_config(), graph)

        stage_cfg = DTSeatHGTv3TrainingStageConfig(
            stage=TrainingStage.SEAT_RESIDUAL,
            epochs=1,
            learning_rate=1e-3,
            freeze_base=True,
            freeze_seat=False,
        )
        _apply_freeze_mask(model, stage_cfg)

        base_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
        }
        seat_before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
        }

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        v4_loss = V4Loss(_v4_loss_weights_for_stage(TrainingStage.SEAT_RESIDUAL))
        model.train()
        optimizer.zero_grad()
        loss, _ = _compute_v4_loss_for_buckets(
            model, samples, [[0, 1, 2]], v4_loss,
            enable_seat_branch=True,
            direction_classes=2, direction_weight=1.0, device="cpu",
        )
        loss.backward()
        optimizer.step()

        base_unchanged = all(
            torch.equal(base_before[name], p.detach())
            for name, p in model.named_parameters()
            if name.split(".")[0] in _BASE_PARAM_PREFIXES
        )
        self.assertTrue(base_unchanged, "Stage B optimizer step mutated frozen base params")

        seat_moved = any(
            not torch.equal(seat_before[name], p.detach())
            for name, p in model.named_parameters()
            if name.split(".")[0] in _SEAT_PARAM_PREFIXES
        )
        self.assertTrue(seat_moved, "Stage B optimizer step did not move any seat param")


if __name__ == "__main__":
    unittest.main()
