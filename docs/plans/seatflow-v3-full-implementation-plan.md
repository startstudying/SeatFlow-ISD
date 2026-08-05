# SeatFlow V3 Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零开发完整 SeatFlow V3，交付 Point-in-Time 数据集、SeatFlow-FDG 模型、严格评估、回测、模拟盘、Python Research Engine、Spring Boot Control Center、Next.js Research Console 和可部署基础设施。

**Architecture:** 系统采用三个正式应用：Python Research Engine 负责全部金融研究语义，Spring Boot Control Center 负责权限、任务、版本、审批和审计，Next.js Research Console 负责操作与可视化。研究数据采用 Parquet + DuckDB + MinIO，控制面采用 PostgreSQL，Redis 用于队列、缓存和实时进度，MLflow 用于实验与模型注册。

**Tech Stack:** Python 3.11、Pydantic 2、Polars/Pandas、DuckDB、PyArrow、PyTorch、PyTorch Geometric、FastAPI、MLflow、Java 21、Spring Boot 3、PostgreSQL、Redis、MinIO、Next.js 15、React 19、TypeScript、React Query、ECharts、Docker Compose、Prometheus、Grafana。

## Global Constraints

- V3 从零设计，旧 V2 文档和旧 DT-SeatHGT 不作为实现底座。
- `Cycle` 是 Train、Validation、Test 和 Blind 的切分原子。
- 所有正式输入满足 `available_time <= sample.cutoff_time`。
- 未成熟标签不得进入席位历史画像。
- 股票、席位、事件和关系 ID 或 ID Hash 不得作为连续特征。
- Base Residual 必须由严格 Walk-Forward OOF 预测构建。
- 正式训练必须验证 Dataset Seal。
- Relation State 必须直接控制边激活、邻居选择、消息和注意力权重。
- Direct 与 Spillover 必须独立构建 Universe、Loss、指标和门禁。
- Python 是标签、Cycle、图、模型、指标、回测和模拟盘信号的唯一实现端。
- PostgreSQL 是任务与审批状态的权威来源，Redis 不是唯一状态源。
- 不得使用合成结果伪装正式指标、预测或回测结果。
- 不得因 OOM 静默缩短数据范围、减少 Fold、Seed、Blind 或消融项。
- 每个生产行为变更必须先有失败测试，再写最小实现。
- 每个任务必须独立提交；未通过任务验收不得进入依赖任务。

---

## Program Execution Order

```text
M0  仓库安全、契约与三应用骨架
M1  Bronze 数据采集与血缘
M2  席位身份、Event 与 Cycle
M3  特征、标签、Sample、Split 与 Dataset Seal
M4  Base Model 与严格 OOF
M5  Seat Memory、Event Encoder 与 Direct
M6  Propagation Candidate、Relation State 与 Spillover
M7  SeatFlow-FDG 联合训练
M8  Walk-Forward、消融、Placebo 与 Blind
M9  回测、成本与模拟盘内核
M10 Python API 与 Worker
M11 Java Control Center
M12 Next.js Research Console
M13 基础设施、监控与部署
M14 全链路 E2E 与发布门禁
M15 Blind Evaluation、模型 Promotion 与每日推理
```

前端、Java 和基础设施可与数据模型轨道并行，但只能基于冻结契约，不能生成虚假研究结果。

---

### Task 1: 建立 V3 安全边界和仓库骨架

**Files:**
- Create: `legacy-reference/README.md`
- Create: `research-engine/pyproject.toml`
- Create: `research-engine/src/seatflow/__init__.py`
- Create: `research-engine/tests/test_package_boundary.py`
- Create: `control-center/pom.xml`
- Create: `control-center/src/main/java/com/seatflow/SeatFlowApplication.java`
- Create: `control-center/src/test/java/com/seatflow/SeatFlowApplicationTests.java`
- Create: `research-console/package.json`
- Create: `research-console/tsconfig.json`
- Create: `research-console/app/page.tsx`
- Create: `research-console/tests/smoke.test.tsx`
- Create: `.github/workflows/v3-ci.yml`

**Interfaces:**
- Produces: Python package `seatflow`, Java application `SeatFlowApplication`, Next.js root page, CI entry points.
- Consumes: branch `rewrite/seatflow-v3` and frozen V3 architecture.

- [ ] **Step 1: Create immutable legacy tag**

Run:

```bash
git tag -a legacy-before-seatflow-v3 9d7cda4a2bce1a11f6e125475de938666cdd7140 -m "Legacy snapshot before SeatFlow V3 rewrite"
git push origin legacy-before-seatflow-v3
```

Expected: tag points to the pre-V3 main commit.

- [ ] **Step 2: Write Python boundary test**

```python
from pathlib import Path


def test_v3_package_does_not_import_legacy_src() -> None:
    root = Path(__file__).resolve().parents[2]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "research-engine" / "src").rglob("*.py")
    )
    assert "src.seatflow" not in source
    assert "backend-java" not in source
```

- [ ] **Step 3: Run the boundary test and verify RED**

Run: `pytest research-engine/tests/test_package_boundary.py -v`

Expected: FAIL because the V3 package and files do not exist.

- [ ] **Step 4: Create minimal Python, Java and Next.js skeletons**

Python package must expose:

```python
__version__ = "3.0.0-dev"
```

Java application must start with `SpringApplication.run` and expose Actuator health. Next.js root page must render `SeatFlow V3` and `No formal model is active`.

- [ ] **Step 5: Add CI commands**

CI runs:

```bash
python -m pytest research-engine/tests -q
mvn -f control-center/pom.xml test
npm --prefix research-console ci
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console run test
npm --prefix research-console run build
```

- [ ] **Step 6: Verify GREEN**

Run all commands above. Expected: all pass, and no formal model claim appears.

- [ ] **Step 7: Commit**

```bash
git add legacy-reference research-engine control-center research-console .github/workflows/v3-ci.yml
git commit -m "build: initialize SeatFlow V3 applications"
```

---

### Task 2: 冻结跨语言领域契约

**Files:**
- Create: `contracts/json-schema/seat.schema.json`
- Create: `contracts/json-schema/stock.schema.json`
- Create: `contracts/json-schema/lhb-event.schema.json`
- Create: `contracts/json-schema/seat-participation.schema.json`
- Create: `contracts/json-schema/event-cycle.schema.json`
- Create: `contracts/json-schema/prediction-sample.schema.json`
- Create: `contracts/json-schema/propagation-candidate.schema.json`
- Create: `contracts/json-schema/task-status.schema.json`
- Create: `contracts/json-schema/error.schema.json`
- Create: `contracts/json-schema/artifact-reference.schema.json`
- Create: `contracts/tests/test_json_schemas.py`

**Interfaces:**
- Produces: JSON Schema 2020-12 domain contracts with `$id` under `https://seatflow.local/schemas/v3/`.
- Consumes: field definitions from `docs/architecture/seatflow-v3-design.md`.

- [ ] **Step 1: Write schema validation tests**

```python
import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path("contracts/json-schema")


def test_all_schemas_are_draft_2020_12() -> None:
    for path in SCHEMA_DIR.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest contracts/tests/test_json_schemas.py -v`

Expected: FAIL because schemas do not exist.

- [ ] **Step 3: Implement schemas**

Every financial record containing a business time must define `availableTime` as an ISO-8601 timestamp. IDs are strings. Monetary values use numbers plus explicit currency convention in description. Enums are frozen in schema, including `seatType`, `taskStatus`, `tradeabilityState` and `modelValidity`.

- [ ] **Step 4: Add forbidden numeric identity test**

```python
def test_identity_fields_are_strings() -> None:
    for name in ["seat", "stock", "lhb-event", "event-cycle", "prediction-sample"]:
        schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
        for field, definition in schema["properties"].items():
            if field.lower().endswith("id"):
                assert definition["type"] == "string"
```

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest contracts/tests/test_json_schemas.py -v
git add contracts
git commit -m "feat(contracts): freeze SeatFlow V3 domain schemas"
```

---

### Task 3: 冻结 OpenAPI、任务事件和状态机

**Files:**
- Create: `contracts/openapi/control-api.yaml`
- Create: `contracts/openapi/research-api.yaml`
- Create: `contracts/events/task-events.schema.json`
- Create: `contracts/tests/test_openapi_contracts.py`
- Create: `docs/api/v3-contract-rules.md`

**Interfaces:**
- Produces: `/api/v1` Control API, `/research/v1` Research API, SSE task event schema.
- Consumes: error and artifact reference schemas from Task 2.

- [ ] **Step 1: Write contract tests**

Tests must assert that both OpenAPI documents parse, every operation has an `operationId`, every error references `SeatFlowError`, and long-running operations return a `taskId` rather than inline large payloads.

- [ ] **Step 2: Verify RED**

Run: `pytest contracts/tests/test_openapi_contracts.py -v`

Expected: FAIL because OpenAPI files are missing.

- [ ] **Step 3: Implement required endpoints**

Control API groups:

```text
/auth
/projects
/data-sources
/datasets
/quality-gates
/experiments
/runs
/models
/predictions
/backtests
/paper-trading
/audit
/system
```

Research API groups:

```text
/health
/tasks
/datasets/build
/datasets/validate
/training/base
/training/direct
/training/spillover
/training/full
/evaluation/walk-forward
/evaluation/blind
/backtests
/inference/daily
```

- [ ] **Step 4: Freeze task state transitions**

```text
REQUESTED -> QUEUED -> RUNNING -> COMPLETED
REQUESTED/QUEUED/RUNNING -> CANCELLED
RUNNING -> FAILED
REQUESTED/QUEUED/RUNNING -> BLOCKED
```

No transition may leave a terminal state.

- [ ] **Step 5: Verify and commit**

```bash
pytest contracts/tests/test_openapi_contracts.py -v
git add contracts docs/api/v3-contract-rules.md
git commit -m "feat(contracts): add V3 APIs and task events"
```

---

### Task 4: 建立本地基础设施和健康契约

**Files:**
- Create: `infra/compose/docker-compose.yml`
- Create: `infra/compose/.env.example`
- Create: `infra/database/postgres/init.sql`
- Create: `infra/monitoring/prometheus.yml`
- Create: `infra/monitoring/grafana/provisioning/datasources/datasource.yml`
- Create: `infra/tests/test_compose_config.py`

**Interfaces:**
- Produces: PostgreSQL, Redis, MinIO, MLflow, Prometheus and Grafana services.
- Consumes: fixed service names `postgres`, `redis`, `minio`, `mlflow`, `prometheus`, `grafana`.

- [ ] **Step 1: Write Compose structure test**

Test parses YAML and asserts services, health checks, named volumes, non-root secret injection and no committed real credential values.

- [ ] **Step 2: Verify RED**

Run: `pytest infra/tests/test_compose_config.py -v`

- [ ] **Step 3: Implement Compose**

Required ports are configurable through environment variables. `.env.example` contains development-only placeholder values. MinIO creates buckets `datasets`, `models`, `reports`, `predictions`, `backtests` through an idempotent init container.

- [ ] **Step 4: Verify configuration**

```bash
docker compose -f infra/compose/docker-compose.yml config
pytest infra/tests/test_compose_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add infra
git commit -m "build(infra): add SeatFlow V3 local platform"
```

---

### Task 5: 实现 Bronze 记录、Manifest 和数据源接口

**Files:**
- Create: `research-engine/src/seatflow/domain/time.py`
- Create: `research-engine/src/seatflow/ingestion/models.py`
- Create: `research-engine/src/seatflow/ingestion/provider.py`
- Create: `research-engine/src/seatflow/ingestion/manifest.py`
- Create: `research-engine/src/seatflow/ingestion/parquet_writer.py`
- Create: `research-engine/tests/ingestion/test_bronze_records.py`
- Create: `research-engine/tests/ingestion/test_manifest.py`

**Interfaces:**
- Produces: `BronzeRecord`, `AvailabilityPolicy`, `DataProvider`, `IngestionManifest`, `write_bronze_partition`.
- `DataProvider.fetch_*` returns `Iterable[BronzeRecord]`.

- [ ] **Step 1: Write failing Bronze tests**

Tests assert `event_time`, `available_time`, `ingested_at`, `source`, `source_record_id`, `source_version`, `revision_no`, `raw_payload_hash`, `ingestion_run_id` and `source_confidence` are mandatory.

- [ ] **Step 2: Verify RED**

Run: `pytest research-engine/tests/ingestion -v`

- [ ] **Step 3: Implement immutable models**

Use timezone-aware UTC `datetime`. Reject `available_time < event_time` only when the provider policy forbids pre-announced records; always reject naive timestamps.

- [ ] **Step 4: Implement deterministic manifest hashing**

Manifest hash is SHA-256 over canonical JSON with sorted keys and normalized timestamps.

- [ ] **Step 5: Verify GREEN and commit**

```bash
pytest research-engine/tests/ingestion -v
git add research-engine/src/seatflow/ingestion research-engine/src/seatflow/domain/time.py research-engine/tests/ingestion
git commit -m "feat(data): add Bronze contracts and manifests"
```

---

### Task 6: 实现多数据源适配器和冲突报告

**Files:**
- Create: `research-engine/src/seatflow/ingestion/providers/tushare.py`
- Create: `research-engine/src/seatflow/ingestion/providers/akshare.py`
- Create: `research-engine/src/seatflow/ingestion/providers/file_provider.py`
- Create: `research-engine/src/seatflow/ingestion/reconcile.py`
- Create: `research-engine/tests/ingestion/test_provider_contract.py`
- Create: `research-engine/tests/ingestion/test_reconcile.py`

**Interfaces:**
- Produces: provider capability probe, daily bars, stock master, trade calendar, LHB summary, LHB participation, memberships and tradeability fetch methods.
- Produces: `ConflictReport` and `ReconciledRecord`.

- [ ] **Step 1: Write provider contract tests**

Use a fake transport, not live network. Tests verify every returned record retains provider and availability evidence.

- [ ] **Step 2: Verify RED**

Run: `pytest research-engine/tests/ingestion/test_provider_contract.py -v`

- [ ] **Step 3: Implement adapters behind injected transports**

No adapter may read environment variables at import time. Missing permission returns capability `DEGRADED` with a machine-readable reason.

- [ ] **Step 4: Write and implement conflict tests**

Conflicting amounts from two sources must preserve both Bronze records, select one Silver value through a versioned rule, and produce a report containing source values, rule version and confidence.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/ingestion -v
git add research-engine/src/seatflow/ingestion/providers research-engine/src/seatflow/ingestion/reconcile.py research-engine/tests/ingestion
git commit -m "feat(data): add source adapters and reconciliation"
```

---

### Task 7: 实现股票与席位身份 Point-in-Time 归一

**Files:**
- Create: `research-engine/src/seatflow/identity/stock_identity.py`
- Create: `research-engine/src/seatflow/identity/seat_identity.py`
- Create: `research-engine/src/seatflow/identity/normalizer.py`
- Create: `research-engine/src/seatflow/identity/repository.py`
- Create: `research-engine/tests/identity/test_stock_identity.py`
- Create: `research-engine/tests/identity/test_seat_identity.py`

**Interfaces:**
- Produces: `resolve_stock(raw_code, at_time) -> StockIdentity`.
- Produces: `resolve_seat(raw_name, at_time) -> SeatIdentityResolution`.

- [ ] **Step 1: Write historical identity tests**

A renamed seat must resolve to the historical canonical identity valid at the event time. The current alias must not overwrite past records.

- [ ] **Step 2: Verify RED**

Run: `pytest research-engine/tests/identity -v`

- [ ] **Step 3: Implement normalization pipeline**

Normalize whitespace, punctuation, branch suffixes and known aliases. Return confidence and evidence. Unresolved seats receive a stable `UNKNOWN` identity generated from provider-scoped raw identity, not a numeric continuous hash.

- [ ] **Step 4: Add collision and interval tests**

Reject overlapping validity intervals for the same alias when they resolve to different canonical seats without explicit conflict status.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/identity -v
git add research-engine/src/seatflow/identity research-engine/tests/identity
git commit -m "feat(identity): add PIT stock and seat resolution"
```

---

### Task 8: 实现 LHBEvent、SeatParticipation 与 EventCycle

**Files:**
- Create: `research-engine/src/seatflow/dataset/events.py`
- Create: `research-engine/src/seatflow/dataset/cycles.py`
- Create: `research-engine/src/seatflow/dataset/calendar.py`
- Create: `research-engine/tests/dataset/test_events.py`
- Create: `research-engine/tests/dataset/test_cycles.py`

**Interfaces:**
- Produces: `build_events(records) -> list[LHBEvent]`.
- Produces: `assign_cycles(events, trading_calendar) -> list[EventCycle]`.

- [ ] **Step 1: Write event aggregation test**

One event with five buy seats and five sell seats must preserve ten participation rows and one event row.

- [ ] **Step 2: Write cycle extension test**

Given an event on trading day D0, initial end is D0+5 trading days. A repeat event on D3 extends end to D3+5. An event after the extended end creates a new cycle.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/dataset/test_events.py research-engine/tests/dataset/test_cycles.py -v`

- [ ] **Step 4: Implement minimal event and cycle builders**

Cycle IDs are deterministic from stock ID and first event date. Calendar arithmetic uses exchange trading days, never calendar-day addition.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/dataset/test_events.py research-engine/tests/dataset/test_cycles.py -v
git add research-engine/src/seatflow/dataset research-engine/tests/dataset
git commit -m "feat(dataset): build LHB events and event cycles"
```

---

### Task 9: 实现 Direct、Spillover 和 Base Universe

**Files:**
- Create: `research-engine/src/seatflow/dataset/universe.py`
- Create: `research-engine/src/seatflow/dataset/propagation_candidates.py`
- Create: `research-engine/tests/dataset/test_universe.py`
- Create: `research-engine/tests/dataset/test_propagation_candidates.py`

**Interfaces:**
- Produces: `build_direct_pool`.
- Produces: `build_spillover_pool`.
- Produces: `build_base_pool`.
- Produces: `PropagationCandidateBuilder.build(cutoff_time, source_events)`.

- [ ] **Step 1: Write pool separation tests**

A stock directly on the LHB at prediction time cannot appear as a Spillover target for the same sample date. Context stocks may pass messages without participating in Direct Loss.

- [ ] **Step 2: Verify RED**

Run: `pytest research-engine/tests/dataset/test_universe.py -v`

- [ ] **Step 3: Implement candidate relations**

Implement same-seat, same-coalition, same-industry, same-concept, capital-rotation and auxiliary comovement candidates using only records available by cutoff.

- [ ] **Step 4: Add deterministic Top-K test**

Candidate ordering uses score descending and target stock ID as stable tie-breaker. Every relation type has an explicit limit.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/dataset/test_universe.py research-engine/tests/dataset/test_propagation_candidates.py -v
git add research-engine/src/seatflow/dataset research-engine/tests/dataset
git commit -m "feat(dataset): add Direct, Spillover and Base universes"
```

---

### Task 10: 实现特征注册表和 PIT 特征构建

**Files:**
- Create: `research-engine/src/seatflow/features/registry.py`
- Create: `research-engine/src/seatflow/features/stock.py`
- Create: `research-engine/src/seatflow/features/seat.py`
- Create: `research-engine/src/seatflow/features/event.py`
- Create: `research-engine/src/seatflow/features/theme.py`
- Create: `research-engine/src/seatflow/features/market.py`
- Create: `research-engine/src/seatflow/features/scaling.py`
- Create: `research-engine/tests/features/test_registry.py`
- Create: `research-engine/tests/features/test_pit_features.py`
- Create: `research-engine/tests/features/test_identity_leakage.py`

**Interfaces:**
- Produces: `FeatureDefinition`, `FeatureRegistry`, entity-specific builders and fold-fitted scalers.

- [ ] **Step 1: Write registry completeness test**

Every feature declares entity type, source tables, lookback, availability rule, maturity rule, transform, normalization, missing policy, dtype, continuous/discrete flag, train permission and lineage version.

- [ ] **Step 2: Write identity leakage test**

```python
def test_continuous_features_exclude_identity_fields(feature_registry):
    forbidden = {"stock_id", "seat_id", "event_id", "cycle_id", "id_hash"}
    continuous = {f.name for f in feature_registry if f.kind == "continuous"}
    assert continuous.isdisjoint(forbidden)
```

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/features -v`

- [ ] **Step 4: Implement builders**

Stock uses 60-day history; Seat uses 5/20/60/120-day mature history; graph uses 20-day event context. Event anomaly features use `log1p`, fold-fitted robust scaling and fold-fitted clipping without ordinary denoising.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/features -v
git add research-engine/src/seatflow/features research-engine/tests/features
git commit -m "feat(features): add PIT feature registry and builders"
```

---

### Task 11: 实现五日路径、累计标签和可交易状态

**Files:**
- Create: `research-engine/src/seatflow/labels/returns.py`
- Create: `research-engine/src/seatflow/labels/tradeability.py`
- Create: `research-engine/src/seatflow/labels/builder.py`
- Create: `research-engine/tests/labels/test_returns.py`
- Create: `research-engine/tests/labels/test_tradeability.py`

**Interfaces:**
- Produces: `build_return_path` and `compound_horizons`.
- Produces: `classify_tradeability`.

- [ ] **Step 1: Write exact formula tests**

Use fixed prices and assert R1, R3 and R5 exactly match compounding from the daily path within `1e-12`.

- [ ] **Step 2: Write execution-state tests**

Cover `VALID_TRADABLE`, `SUSPENDED_AT_ENTRY`, `ONE_PRICE_LIMIT_UP`, `ONE_PRICE_LIMIT_DOWN`, `MISSING_ENTRY_PRICE`, `DELISTED_DURING_HORIZON` and `CORPORATE_ACTION_SPECIAL`.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/labels -v`

- [ ] **Step 4: Implement labels**

Entry is t+1 open. Exits are t+1, t+3 and t+5 close. Store raw, market-excess, industry-excess, direction, tail-risk and tradeability labels.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/labels -v
git add research-engine/src/seatflow/labels research-engine/tests/labels
git commit -m "feat(labels): add multi-horizon return paths"
```

---

### Task 12: 实现 Sample、Cycle 原子切分和 Dataset Seal

**Files:**
- Create: `research-engine/src/seatflow/dataset/samples.py`
- Create: `research-engine/src/seatflow/dataset/splits.py`
- Create: `research-engine/src/seatflow/dataset/quality.py`
- Create: `research-engine/src/seatflow/dataset/seal.py`
- Create: `research-engine/src/seatflow/dataset/manifest.py`
- Create: `research-engine/tests/dataset/test_samples.py`
- Create: `research-engine/tests/dataset/test_cycle_atomic_split.py`
- Create: `research-engine/tests/dataset/test_seal.py`

**Interfaces:**
- Produces: `PredictionSample`, `WalkForwardFold`, `QualityGateReport`, `DatasetManifest`, `DatasetSeal`.

- [ ] **Step 1: Write rolling sample test**

Every trading day in an active Cycle generates one sample until cycle end. Non-trading days generate none.

- [ ] **Step 2: Write atomic split test**

Assert each `cycle_id` appears in exactly one of train, validation, embargo, test or blind.

- [ ] **Step 3: Write fail-closed Seal test**

One PIT violation or one cross-split Cycle prevents Seal creation and sets `training_allowed=false`.

- [ ] **Step 4: Verify RED**

Run: `pytest research-engine/tests/dataset -v`

- [ ] **Step 5: Implement sample build, quality gates and canonical manifest hash**

- [ ] **Step 6: Verify and commit**

```bash
pytest research-engine/tests/dataset -v
git add research-engine/src/seatflow/dataset research-engine/tests/dataset
git commit -m "feat(dataset): seal PIT samples and atomic splits"
```

---

### Task 13: 实现图 Schema、快照和 PyG 张量化

**Files:**
- Create: `research-engine/src/seatflow/graph/schema.py`
- Create: `research-engine/src/seatflow/graph/snapshot.py`
- Create: `research-engine/src/seatflow/graph/tensorize.py`
- Create: `research-engine/src/seatflow/graph/topk.py`
- Create: `research-engine/tests/graph/test_schema.py`
- Create: `research-engine/tests/graph/test_tensorize.py`
- Create: `research-engine/tests/graph/test_topk.py`

**Interfaces:**
- Produces: `GraphSchema`, `GraphSnapshot`, `tensorize_snapshot -> HeteroData`.

- [ ] **Step 1: Write node and edge schema tests**

Required nodes: Seat, Event, Stock, Theme. Required relations match the V3 architecture. Market is graph-level context.

- [ ] **Step 2: Write edge feature safety test**

Reject edge columns containing numeric identity or absolute-date identity proxies.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/graph -v`

- [ ] **Step 4: Implement stable local indexes and masks**

`target_mask`, `context_mask`, `loss_mask` and `backtest_mask` must be separate tensors.

- [ ] **Step 5: Add shape and round-trip tests**

Serialize snapshot to Parquet/JSON manifest, reload and tensorize to identical IDs, features and relation counts.

- [ ] **Step 6: Verify and commit**

```bash
pytest research-engine/tests/graph -v
git add research-engine/src/seatflow/graph research-engine/tests/graph
git commit -m "feat(graph): add V3 graph snapshots and tensorization"
```

---

### Task 14: 实现 Market Base Encoder 和严格 OOF

**Files:**
- Create: `research-engine/src/seatflow/models/base/encoder.py`
- Create: `research-engine/src/seatflow/models/base/heads.py`
- Create: `research-engine/src/seatflow/training/base_trainer.py`
- Create: `research-engine/src/seatflow/training/oof.py`
- Create: `research-engine/tests/models/test_base_shapes.py`
- Create: `research-engine/tests/training/test_oof_independence.py`
- Create: `research-engine/tests/training/test_base_micro_overfit.py`

**Interfaces:**
- Produces: `MarketBaseEncoder.forward(stock_sequence, market_context)`.
- Produces: `BasePrediction` with path[5], R1, R3, R5 and regime embedding.
- Produces: `generate_walk_forward_oof`.

- [ ] **Step 1: Write shape and information-boundary tests**

Base model input types must not accept Seat, Event or graph tensors.

- [ ] **Step 2: Write OOF independence test**

For every prediction row, assert its cycle is not in the fitted training cycles for that fold.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/models/test_base_shapes.py research-engine/tests/training/test_oof_independence.py -v`

- [ ] **Step 4: Implement encoder, path heads and fold trainer**

Use compact TCN/Transformer encoder configurable for 8GB GPU. Horizon returns are derived from path compounding, not independent inconsistent outputs.

- [ ] **Step 5: Run micro-overfit**

Run: `pytest research-engine/tests/training/test_base_micro_overfit.py -v`

Expected: model overfits a deterministic tiny batch and loss decreases by at least 95%.

- [ ] **Step 6: Commit**

```bash
git add research-engine/src/seatflow/models/base research-engine/src/seatflow/training research-engine/tests/models research-engine/tests/training
git commit -m "feat(model): add Base encoder and strict OOF"
```

---

### Task 15: 实现 Seat Memory、Event Encoder 和 Direct Decoder

**Files:**
- Create: `research-engine/src/seatflow/models/seat/memory.py`
- Create: `research-engine/src/seatflow/models/event/interaction_encoder.py`
- Create: `research-engine/src/seatflow/models/direct/decoder.py`
- Create: `research-engine/src/seatflow/models/direct/model.py`
- Create: `research-engine/src/seatflow/training/direct_trainer.py`
- Create: `research-engine/tests/models/test_seat_memory.py`
- Create: `research-engine/tests/models/test_event_encoder.py`
- Create: `research-engine/tests/models/test_direct_model.py`
- Create: `research-engine/tests/training/test_direct_micro_overfit.py`

**Interfaces:**
- Produces: `SeatBehaviorMemory`, `EventInteractionEncoder`, `DirectEffectDecoder`, `DirectModel`.

- [ ] **Step 1: Write maturity-mask test**

Seat history records whose `label_end_time` exceeds cutoff must not affect the output embedding.

- [ ] **Step 2: Write permutation test**

Reordering seat participation rows without changing ranks and values must preserve pooled event representation.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/models/test_seat_memory.py research-engine/tests/models/test_event_encoder.py -v`

- [ ] **Step 4: Implement Direct path and residual heads**

Outputs include direct path[5], R1/R3/R5, continuation, reversal and uncertainty.

- [ ] **Step 5: Add baseline adapters**

Support Amount Only, Seat Type, Anonymous Seat, Real Seat and Shuffled Seat using one shared trainer and explicit config.

- [ ] **Step 6: Run tests and commit**

```bash
pytest research-engine/tests/models research-engine/tests/training/test_direct_micro_overfit.py -v
git add research-engine/src/seatflow/models research-engine/src/seatflow/training/direct_trainer.py research-engine/tests/models research-engine/tests/training
git commit -m "feat(model): add Seat Event Direct pathway"
```

---

### Task 16: 实现 Relation State Gate 和 Spillover Decoder

**Files:**
- Create: `research-engine/src/seatflow/models/flow/relation_state.py`
- Create: `research-engine/src/seatflow/models/flow/gated_message_passing.py`
- Create: `research-engine/src/seatflow/models/spillover/decoder.py`
- Create: `research-engine/src/seatflow/models/spillover/model.py`
- Create: `research-engine/src/seatflow/training/spillover_trainer.py`
- Create: `research-engine/tests/models/test_relation_state.py`
- Create: `research-engine/tests/models/test_gated_message_passing.py`
- Create: `research-engine/tests/models/test_spillover_model.py`

**Interfaces:**
- Produces: `RelationStateGate.forward(edge_features, age, market_regime)`.
- Produces: `GatedHeteroMessagePassing`.
- Produces: `SpilloverModel`.

- [ ] **Step 1: Write routing-effect test**

With activation gate zero, the candidate edge must contribute exactly zero message. With gate one and non-zero message parameters, output must change.

- [ ] **Step 2: Write Top-K activation test**

Only the configured highest activated candidates per relation remain in the message graph.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/models/test_relation_state.py research-engine/tests/models/test_gated_message_passing.py -v`

- [ ] **Step 4: Implement sparse relation state and direction gate**

Activation uses sigmoid, strength uses non-negative transform, direction uses tanh, reliability multiplies the final edge weight.

- [ ] **Step 5: Implement Spillover outputs and isolated metrics**

Spillover loss only applies to Spillover Pool. Direct targets are masked out.

- [ ] **Step 6: Verify and commit**

```bash
pytest research-engine/tests/models/test_relation_state.py research-engine/tests/models/test_gated_message_passing.py research-engine/tests/models/test_spillover_model.py -v
git add research-engine/src/seatflow/models/flow research-engine/src/seatflow/models/spillover research-engine/src/seatflow/training/spillover_trainer.py research-engine/tests/models
git commit -m "feat(model): add gated flow and Spillover pathway"
```

---

### Task 17: 实现 SeatFlow-FDG 联合模型与损失

**Files:**
- Create: `research-engine/src/seatflow/models/fdg/model.py`
- Create: `research-engine/src/seatflow/models/fdg/fusion.py`
- Create: `research-engine/src/seatflow/models/fdg/output.py`
- Create: `research-engine/src/seatflow/training/losses.py`
- Create: `research-engine/src/seatflow/training/fdg_trainer.py`
- Create: `research-engine/tests/models/test_fdg_output.py`
- Create: `research-engine/tests/models/test_fdg_gradients.py`
- Create: `research-engine/tests/training/test_loss_masks.py`
- Create: `research-engine/tests/training/test_fdg_micro_overfit.py`

**Interfaces:**
- Produces: `SeatFlowFDG.forward(batch) -> SeatFlowPredictionBatch`.
- Produces: `SeatFlowLoss.forward(prediction, target, masks)`.

- [ ] **Step 1: Write additive decomposition test**

Assert `final_path == base_path + direct_path + spillover_path`, and cumulative R1/R3/R5 exactly compound from final path.

- [ ] **Step 2: Write gradient coverage test**

A combined loss backward pass must produce finite gradients in Base, Seat Memory, Event Encoder, Relation State, Direct and Spillover parameters.

- [ ] **Step 3: Write mask isolation test**

Direct rows do not enter Spillover loss; context-only rows do not enter Direct loss; non-tradable rows may enter event-study loss but not backtest loss.

- [ ] **Step 4: Verify RED**

Run: `pytest research-engine/tests/models/test_fdg_output.py research-engine/tests/models/test_fdg_gradients.py research-engine/tests/training/test_loss_masks.py -v`

- [ ] **Step 5: Implement model, fusion and loss terms**

Include return, ranking, direction, relation activation/direction/strength, spillover, calibration and risk losses.

- [ ] **Step 6: Run full model micro-overfit and commit**

```bash
pytest research-engine/tests/training/test_fdg_micro_overfit.py -v
git add research-engine/src/seatflow/models/fdg research-engine/src/seatflow/training research-engine/tests/models research-engine/tests/training
git commit -m "feat(model): assemble SeatFlow FDG"
```

---

### Task 18: 实现 Walk-Forward、指标、消融和 Placebo

**Files:**
- Create: `research-engine/src/seatflow/evaluation/metrics.py`
- Create: `research-engine/src/seatflow/evaluation/walk_forward.py`
- Create: `research-engine/src/seatflow/evaluation/ablations.py`
- Create: `research-engine/src/seatflow/evaluation/placebo.py`
- Create: `research-engine/src/seatflow/evaluation/calibration.py`
- Create: `research-engine/tests/evaluation/test_metrics.py`
- Create: `research-engine/tests/evaluation/test_walk_forward.py`
- Create: `research-engine/tests/evaluation/test_placebo.py`

**Interfaces:**
- Produces: fold reports, aggregate reports and validity evidence.

- [ ] **Step 1: Write metric reference tests**

Use fixed arrays to validate MAE, Huber, AUC, IC, RankIC, NDCG@K, Edge AUPR, Brier and ECE against known values.

- [ ] **Step 2: Write Walk-Forward boundary test**

Assert chronological order, at least 5 trading-day embargo and cycle atomicity.

- [ ] **Step 3: Write placebo reproducibility test**

Shuffled Seat and Shuffled Graph preserve row counts and degree constraints while changing semantic assignments deterministically for a seed.

- [ ] **Step 4: Verify RED and implement**

Run: `pytest research-engine/tests/evaluation -v`

- [ ] **Step 5: Implement the ablation matrix**

Required variants: Base, Amount Only, Seat Type, Anonymous Seat, Real Seat Direct, No Graph, Shuffled Graph, Real Graph Spillover, Full FDG and Full FDG without Relation State routing.

- [ ] **Step 6: Commit**

```bash
git add research-engine/src/seatflow/evaluation research-engine/tests/evaluation
git commit -m "feat(evaluation): add walk-forward and evidence matrix"
```

---

### Task 19: 实现 Blind Seal 与模型有效性门禁

**Files:**
- Create: `research-engine/src/seatflow/evaluation/blind.py`
- Create: `research-engine/src/seatflow/evaluation/model_gate.py`
- Create: `research-engine/tests/evaluation/test_blind_access.py`
- Create: `research-engine/tests/evaluation/test_model_gate.py`

**Interfaces:**
- Produces: `BlindCandidateManifest`, immutable `BlindEvaluationReport`, `ModelGateDecision`.

- [ ] **Step 1: Write blind access tests**

Development mode cannot read Blind labels or metrics. Execution requires an approved candidate manifest whose hashes match dataset, code, config, checkpoint and cost config.

- [ ] **Step 2: Write gate tests**

Failure of Direct, Spillover, economic or calibration gate sets `model_valid=false`, `promotion_allowed=false`, `paper_trading_allowed=false`.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/evaluation/test_blind_access.py research-engine/tests/evaluation/test_model_gate.py -v`

- [ ] **Step 4: Implement immutable reports**

Blind report path includes candidate ID and execution timestamp. Existing report paths cannot be overwritten.

- [ ] **Step 5: Verify and commit**

```bash
pytest research-engine/tests/evaluation -v
git add research-engine/src/seatflow/evaluation research-engine/tests/evaluation
git commit -m "feat(evaluation): seal blind runs and model gates"
```

---

### Task 20: 实现交易成本、回测和模拟盘内核

**Files:**
- Create: `research-engine/src/seatflow/backtest/costs.py`
- Create: `research-engine/src/seatflow/backtest/execution.py`
- Create: `research-engine/src/seatflow/backtest/portfolio.py`
- Create: `research-engine/src/seatflow/backtest/engine.py`
- Create: `research-engine/src/seatflow/backtest/report.py`
- Create: `research-engine/src/seatflow/backtest/paper_account.py`
- Create: `research-engine/tests/backtest/test_costs.py`
- Create: `research-engine/tests/backtest/test_execution.py`
- Create: `research-engine/tests/backtest/test_portfolio.py`
- Create: `research-engine/tests/backtest/test_paper_account.py`

**Interfaces:**
- Produces: `CostModel`, `ExecutionSimulator`, `PortfolioAllocator`, `BacktestEngine`, `PaperAccount`.

- [ ] **Step 1: Write transaction cost tests**

Validate commission minimum, stamp tax on sells, slippage and impact cost with fixed orders.

- [ ] **Step 2: Write unexecutable order tests**

One-price limit-up at entry rejects buy; suspension rejects order; limit-down exit remains pending according to policy.

- [ ] **Step 3: Write overlapping holding-period test**

Top-K signals on consecutive days respect available cash and existing positions.

- [ ] **Step 4: Verify RED and implement**

Run: `pytest research-engine/tests/backtest -v`

- [ ] **Step 5: Enforce model gate in PaperAccount**

Creating a paper session with `paper_trading_allowed=false` raises a non-retryable domain error.

- [ ] **Step 6: Commit**

```bash
git add research-engine/src/seatflow/backtest research-engine/tests/backtest
git commit -m "feat(backtest): add costs execution and paper account"
```

---

### Task 21: 实现 Python Research API、Redis Worker 和产物存储

**Files:**
- Create: `research-engine/src/seatflow/api/app.py`
- Create: `research-engine/src/seatflow/api/routes/health.py`
- Create: `research-engine/src/seatflow/api/routes/tasks.py`
- Create: `research-engine/src/seatflow/api/routes/datasets.py`
- Create: `research-engine/src/seatflow/api/routes/training.py`
- Create: `research-engine/src/seatflow/api/routes/evaluation.py`
- Create: `research-engine/src/seatflow/api/routes/inference.py`
- Create: `research-engine/src/seatflow/tasks/queue.py`
- Create: `research-engine/src/seatflow/tasks/worker.py`
- Create: `research-engine/src/seatflow/tasks/state.py`
- Create: `research-engine/src/seatflow/tasks/artifacts.py`
- Create: `research-engine/tests/api/test_health.py`
- Create: `research-engine/tests/api/test_task_api.py`
- Create: `research-engine/tests/tasks/test_idempotency.py`

**Interfaces:**
- Produces: OpenAPI-compatible Research API and Redis queue messages containing only IDs, versions, URIs, hashes and small parameters.

- [ ] **Step 1: Write health and schema tests**

`/research/v1/health` returns service version and dependency status without secrets.

- [ ] **Step 2: Write idempotency test**

Submitting the same idempotency key and normalized request twice returns the same task ID and does not enqueue a second job.

- [ ] **Step 3: Verify RED**

Run: `pytest research-engine/tests/api research-engine/tests/tasks -v`

- [ ] **Step 4: Implement task state transitions and cancellation**

Worker updates PostgreSQL through the Control Center callback contract or direct internal repository configured for local mode. Redis carries queue and progress events only.

- [ ] **Step 5: Add MinIO artifact writer**

Artifact upload returns URI, SHA-256, size, media type and schema version.

- [ ] **Step 6: Verify and commit**

```bash
pytest research-engine/tests/api research-engine/tests/tasks -v
git add research-engine/src/seatflow/api research-engine/src/seatflow/tasks research-engine/tests/api research-engine/tests/tasks
git commit -m "feat(research-api): add task API worker and artifacts"
```

---

### Task 22: 实现 Spring Boot Control Center 核心

**Files:**
- Create: `control-center/src/main/java/com/seatflow/common/SeatFlowError.java`
- Create: `control-center/src/main/java/com/seatflow/security/SecurityConfig.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskEntity.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskRepository.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskService.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskController.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskEventPublisher.java`
- Create: `control-center/src/main/resources/db/migration/V1__control_core.sql`
- Create: `control-center/src/test/java/com/seatflow/task/TaskServiceTest.java`
- Create: `control-center/src/test/java/com/seatflow/task/TaskControllerTest.java`

**Interfaces:**
- Produces: task creation/query/cancel API, PostgreSQL task state and Redis enqueue/event integration.

- [ ] **Step 1: Write state transition unit tests**

Terminal states reject further transitions. Duplicate idempotency keys return the existing task.

- [ ] **Step 2: Write API contract test**

MockMvc response fields and error model must match `contracts/openapi/control-api.yaml`.

- [ ] **Step 3: Verify RED**

Run: `mvn -f control-center/pom.xml -Dtest=TaskServiceTest,TaskControllerTest test`

- [ ] **Step 4: Implement Flyway schema and services**

Use optimistic locking with a version column. Every write records actor, request ID and trace ID.

- [ ] **Step 5: Verify and commit**

```bash
mvn -f control-center/pom.xml test
git add control-center
git commit -m "feat(control): add task control plane"
```

---

### Task 23: 实现 Control Center 领域模块和审批门禁

**Files:**
- Create packages under: `control-center/src/main/java/com/seatflow/dataset/`
- Create packages under: `control-center/src/main/java/com/seatflow/experiment/`
- Create packages under: `control-center/src/main/java/com/seatflow/model/`
- Create packages under: `control-center/src/main/java/com/seatflow/backtest/`
- Create packages under: `control-center/src/main/java/com/seatflow/paper/`
- Create packages under: `control-center/src/main/java/com/seatflow/audit/`
- Create Flyway migrations: `V2__research_metadata.sql`, `V3__model_and_paper.sql`
- Create tests under corresponding `control-center/src/test/java/com/seatflow/` packages.

**Interfaces:**
- Produces: Dataset Version, Quality Gate, Experiment, Run, Model Version, Promotion, Prediction, Backtest, Paper Session and Audit APIs.

- [ ] **Step 1: Write Quality Gate blocking test**

Invalid Dataset prevents formal experiment creation and returns `DATASET_QUALITY_GATE_FAILED`.

- [ ] **Step 2: Write Promotion blocking test**

Model with `model_valid=false` cannot be promoted.

- [ ] **Step 3: Write paper-trading blocking test**

Paper session requires a promoted model with `paper_trading_allowed=true`.

- [ ] **Step 4: Verify RED**

Run targeted Maven tests for dataset, model and paper packages.

- [ ] **Step 5: Implement modules and audit records**

All approvals record actor, time, old state, new state, reason and evidence URI.

- [ ] **Step 6: Verify and commit**

```bash
mvn -f control-center/pom.xml test
git add control-center
git commit -m "feat(control): add research governance and approvals"
```

---

### Task 24: 实现 SSE、RBAC 和系统监控 API

**Files:**
- Create: `control-center/src/main/java/com/seatflow/security/RbacService.java`
- Create: `control-center/src/main/java/com/seatflow/security/Permission.java`
- Create: `control-center/src/main/java/com/seatflow/task/TaskSseController.java`
- Create: `control-center/src/main/java/com/seatflow/system/SystemStatusController.java`
- Create: `control-center/src/test/java/com/seatflow/security/RbacTest.java`
- Create: `control-center/src/test/java/com/seatflow/task/TaskSseControllerTest.java`

**Interfaces:**
- Produces: resumable SSE with `Last-Event-ID`, role/permission enforcement and dependency status API.

- [ ] **Step 1: Write permission tests**

Separate permissions for dataset seal, blind access, model promotion and paper trading activation.

- [ ] **Step 2: Write SSE replay test**

Reconnect with `Last-Event-ID` returns only subsequent events and then continues streaming.

- [ ] **Step 3: Verify RED and implement**

Run: `mvn -f control-center/pom.xml -Dtest=RbacTest,TaskSseControllerTest test`

- [ ] **Step 4: Verify full Java suite and commit**

```bash
mvn -f control-center/pom.xml test
git add control-center
git commit -m "feat(control): add RBAC SSE and status monitoring"
```

---

### Task 25: 建立 Next.js 设计系统、API Adapter 和真实性状态

**Files:**
- Create: `research-console/components/ui/`
- Create: `research-console/components/layout/`
- Create: `research-console/lib/api/client.ts`
- Create: `research-console/lib/api/generated/`
- Create: `research-console/lib/sse/task-stream.ts`
- Create: `research-console/features/status/model-validity-badge.tsx`
- Create: `research-console/features/status/data-quality-badge.tsx`
- Create: `research-console/tests/api-client.test.ts`
- Create: `research-console/tests/status-badges.test.tsx`

**Interfaces:**
- Produces: typed API client, SSE client, common layout and validity displays.

- [ ] **Step 1: Write API error test**

SeatFlow error responses preserve code, traceId, taskId, retryable and details.

- [ ] **Step 2: Write status truthfulness tests**

Invalid model renders `模型无效` and cannot render a green formal prediction state. Synthetic runs render `非正式运行`.

- [ ] **Step 3: Verify RED**

Run: `npm --prefix research-console test`

- [ ] **Step 4: Implement components and adapters**

Generated client output is isolated and not manually edited. UI components consume a stable adapter layer.

- [ ] **Step 5: Verify and commit**

```bash
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console test
git add research-console
git commit -m "feat(console): add V3 UI foundation and API adapters"
```

---

### Task 26: 实现数据、Dataset、Event/Cycle 和席位页面

**Files:**
- Create: `research-console/app/data-sources/page.tsx`
- Create: `research-console/app/datasets/page.tsx`
- Create: `research-console/app/quality/page.tsx`
- Create: `research-console/app/events/page.tsx`
- Create: `research-console/app/cycles/[cycleId]/page.tsx`
- Create: `research-console/app/seats/page.tsx`
- Create: `research-console/app/seats/[seatId]/page.tsx`
- Create: `research-console/features/data/`
- Create: `research-console/features/cycles/`
- Create: `research-console/features/seats/`
- Create corresponding Vitest tests.

**Interfaces:**
- Consumes: data source, dataset, quality, event, cycle and seat APIs.

- [ ] **Step 1: Write loading, empty, degraded and error-state tests**

Every page must distinguish no data, source degraded, quality failure and request error.

- [ ] **Step 2: Verify RED and implement pages**

Dataset page shows source manifest, feature schema, label policy, split hash, quality gates and Seal state.

- [ ] **Step 3: Verify and commit**

```bash
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console test
git add research-console
git commit -m "feat(console): add data cycle and seat research pages"
```

---

### Task 27: 实现实验、训练、模型比较和图探索页面

**Files:**
- Create: `research-console/app/experiments/page.tsx`
- Create: `research-console/app/experiments/new/page.tsx`
- Create: `research-console/app/runs/[runId]/page.tsx`
- Create: `research-console/app/models/compare/page.tsx`
- Create: `research-console/app/graph/page.tsx`
- Create: `research-console/features/experiments/`
- Create: `research-console/features/training/`
- Create: `research-console/features/graph/`
- Create corresponding tests.

**Interfaces:**
- Consumes: experiment, run, task SSE, evaluation and graph snapshot APIs.

- [ ] **Step 1: Write task SSE reconnect test**

The run page resumes after refresh and does not duplicate metrics when replaying events.

- [ ] **Step 2: Write graph evidence test**

Every displayed path shows horizon, relation type, activation probability, direction, strength, evidence time and source/target nodes.

- [ ] **Step 3: Verify RED and implement**

Model comparison includes Base, Direct, Spillover, Full, Shuffled Seat, No Graph and Shuffled Graph.

- [ ] **Step 4: Verify and commit**

```bash
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console test
git add research-console
git commit -m "feat(console): add experiments training and graph explorer"
```

---

### Task 28: 实现预测、解释、回测、模拟盘和治理页面

**Files:**
- Create: `research-console/app/predictions/page.tsx`
- Create: `research-console/app/predictions/[stockId]/page.tsx`
- Create: `research-console/app/backtests/page.tsx`
- Create: `research-console/app/paper-trading/page.tsx`
- Create: `research-console/app/models/page.tsx`
- Create: `research-console/app/audit/page.tsx`
- Create: `research-console/app/system/page.tsx`
- Create feature packages and tests for each page.

**Interfaces:**
- Consumes: prediction, explanation, backtest, paper, model registry, audit and system APIs.

- [ ] **Step 1: Write prediction decomposition test**

UI displays Base, Direct, Spillover and Final paths, and t+1/t+3/t+5 cumulative values.

- [ ] **Step 2: Write paper-trading gate test**

Activation controls are disabled with the exact gate failure reason when the model is invalid or unpromoted.

- [ ] **Step 3: Write cost visibility test**

Backtest page cannot label a result as formal unless cost configuration and post-cost metrics are present.

- [ ] **Step 4: Verify RED, implement and commit**

```bash
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console test
npm --prefix research-console run build
git add research-console
git commit -m "feat(console): add predictions backtests paper and governance"
```

---

### Task 29: 添加可观测性、镜像和部署配置

**Files:**
- Create: `research-engine/Dockerfile`
- Create: `control-center/Dockerfile`
- Create: `research-console/Dockerfile`
- Modify: `infra/compose/docker-compose.yml`
- Create: `infra/monitoring/grafana/dashboards/seatflow-overview.json`
- Create: `infra/k3s/`
- Create: `research-engine/src/seatflow/observability/tracing.py`
- Create: `research-engine/src/seatflow/observability/metrics.py`
- Create: `control-center/src/main/java/com/seatflow/observability/TraceConfig.java`
- Create: `infra/tests/test_images_and_manifests.py`

**Interfaces:**
- Produces: identical application images for Compose and k3s; trace correlation fields across services.

- [ ] **Step 1: Write deployment manifest tests**

Assert image names, health checks, resource limits, non-root execution and required environment keys are consistent.

- [ ] **Step 2: Verify RED and implement Dockerfiles and k3s manifests**

- [ ] **Step 3: Add metrics**

Include queue depth, task duration, data latency, quality failures, GPU memory, fold/seed progress, API latency and SSE connection count.

- [ ] **Step 4: Verify and commit**

```bash
docker compose -f infra/compose/docker-compose.yml config
pytest infra/tests/test_images_and_manifests.py -v
git add infra research-engine/Dockerfile control-center/Dockerfile research-console/Dockerfile research-engine/src/seatflow/observability control-center/src/main/java/com/seatflow/observability
git commit -m "build: add V3 images observability and deployment"
```

---

### Task 30: 建立合成数据全链路 E2E

**Files:**
- Create: `research-engine/tests/fixtures/synthetic_market.py`
- Create: `tools/e2e/run_synthetic_e2e.py`
- Create: `tools/e2e/assert_e2e_artifacts.py`
- Create: `research-console/e2e/seatflow.spec.ts`
- Create: `.github/workflows/v3-e2e.yml`

**Interfaces:**
- Produces: deterministic end-to-end run across ingestion, dataset, model smoke, API, Java and UI.

- [ ] **Step 1: Create deterministic synthetic fixture**

Fixture contains two seats, three direct stocks, three spillover stocks, repeated events, one cycle extension, one untradable entry and known directional propagation.

- [ ] **Step 2: Write artifact assertions**

Assert creation of source manifest, Dataset Seal, Base OOF, Direct checkpoint, Spillover checkpoint, evaluation report, prediction manifest and UI-visible task completion.

- [ ] **Step 3: Verify RED**

Run: `python tools/e2e/run_synthetic_e2e.py`

- [ ] **Step 4: Implement minimal orchestration required to pass**

The E2E is explicitly marked `SYNTHETIC_NON_FORMAL`; no formal model validity is claimed.

- [ ] **Step 5: Run Playwright and commit**

```bash
python tools/e2e/run_synthetic_e2e.py
npm --prefix research-console run test:e2e
git add tools research-engine/tests/fixtures research-console/e2e .github/workflows/v3-e2e.yml
git commit -m "test(e2e): cover SeatFlow V3 synthetic lifecycle"
```

---

### Task 31: 实现正式发布门禁和 Runbook

**Files:**
- Create: `tools/release/verify_release.py`
- Create: `tools/release/create_release_manifest.py`
- Create: `docs/runbooks/dataset-build.md`
- Create: `docs/runbooks/formal-training.md`
- Create: `docs/runbooks/blind-evaluation.md`
- Create: `docs/runbooks/model-promotion.md`
- Create: `docs/runbooks/paper-trading.md`
- Create: `docs/runbooks/incident-response.md`
- Create: `tools/release/tests/test_verify_release.py`

**Interfaces:**
- Produces: executable release verification and immutable Release Manifest.

- [ ] **Step 1: Write release-blocking tests**

Missing Dataset Seal, failing tests, invalid model gate, missing cost report, uncommitted changes or mismatched Git SHA must block release.

- [ ] **Step 2: Verify RED and implement verifier**

Run: `pytest tools/release/tests/test_verify_release.py -v`

- [ ] **Step 3: Write operational runbooks with exact commands**

Each runbook includes prerequisites, command, expected artifacts, failure codes, rollback and audit evidence.

- [ ] **Step 4: Verify and commit**

```bash
pytest tools/release/tests/test_verify_release.py -v
git add tools/release docs/runbooks
git commit -m "docs: add V3 release gates and runbooks"
```

---

### Task 32: 执行正式候选、Blind、Promotion 和每日推理

**Files:**
- Create: `configs/dataset/formal-v3.yaml`
- Create: `configs/model/seatflow-fdg-v3.yaml`
- Create: `configs/training/formal-8gb.yaml`
- Create: `configs/evaluation/walk-forward-v3.yaml`
- Create: `configs/evaluation/blind-v3.yaml`
- Create: `configs/backtest/cost-v3.yaml`
- Create: `docs/research/v3-formal-evidence-template.md`

**Interfaces:**
- Produces: formal Dataset Version, experiment series, candidate manifest, Blind report, model decision and daily prediction manifest.

- [ ] **Step 1: Build and seal the formal dataset**

Run through Control Center and retain task ID, Dataset Manifest, Seal URI and quality report.

- [ ] **Step 2: Execute Base, Direct, Spillover and Full experiments**

Run all configured folds, seeds, baselines, ablations and placebo variants without silent reductions.

- [ ] **Step 3: Freeze the Blind candidate**

Record Dataset hash, Git SHA, environment hash, config hashes, seed, checkpoint URIs, cost hash and Blind range.

- [ ] **Step 4: Execute Blind once**

Write a new immutable report path. Do not overwrite or retune under the same candidate ID.

- [ ] **Step 5: Evaluate model gates**

Promotion occurs only when all required gates pass. A negative result is recorded as a valid scientific result with `model_valid=false`.

- [ ] **Step 6: Enable daily inference and paper trading only after Promotion**

Every prediction records cutoff, Dataset Version, Model Version, graph snapshot hash, output URI and explanation URI.

- [ ] **Step 7: Commit only configuration and evidence templates**

```bash
git add configs docs/research/v3-formal-evidence-template.md
git commit -m "config: freeze SeatFlow V3 formal protocol"
```

---

## Parallel Work Allocation

- Data team: Tasks 5–13.
- Model team: Tasks 14–20.
- Python platform team: Task 21 and model/data integration.
- Java team: Tasks 22–24.
- Frontend team: Tasks 25–28.
- Platform team: Tasks 1–4, 29–31.
- Research lead: Task 32 and all model/data gate approvals.

Parallel teams share only frozen contracts. A contract change requires schema version bump, compatibility test and explicit review.

---

## Mandatory Checkpoints

### Checkpoint A — M0 complete

- Three applications start.
- Contracts validate.
- Compose configuration validates.
- CI is green.
- No formal prediction exists.

### Checkpoint B — Dataset complete

- PIT, Cycle, label and split tests are green.
- Dataset Seal can be created only for valid synthetic data.
- Data lineage is complete.

### Checkpoint C — Model mechanics complete

- Shape, gradient and micro-overfit tests pass.
- Relation State demonstrably changes routing.
- Base, Direct and Spillover are separable.

### Checkpoint D — Evidence pipeline complete

- Walk-Forward, Placebo, calibration, costs and Blind controls pass tests.
- Model Gate is fail-closed.

### Checkpoint E — Platform complete

- Python, Java and Next.js satisfy contract tests.
- Task creation, progress, cancellation and resume work end to end.
- Invalid models cannot activate paper trading.

### Checkpoint F — Formal evidence complete

- Formal dataset and candidate are frozen.
- Blind executes once.
- Promotion decision is evidence-based and immutable.

---

## Self-Review Results

- Architecture coverage: data, model, evaluation, backtest, Python, Java, frontend, infrastructure, observability, security and release are mapped to Tasks 1–32.
- Placeholder scan: the plan contains no unfinished implementation placeholders.
- Type consistency: task, artifact, Dataset Seal, model gate, prediction and error contracts are defined once and reused across tracks.
- Safety coverage: PIT, Cycle atomicity, mature labels, ID leakage, OOF independence, Blind access, cost visibility and paper-trading gates each have explicit failing tests.

---

## Execution Handoff

Plan complete and saved to `docs/plans/seatflow-v3-full-implementation-plan.md`.

Execution mode is fixed as task-by-task development with fresh context per task, test-first implementation, review after each commit and checkpoint review before the next milestone. ZCode must start at Task 1 and stop at each checkpoint rather than issuing an unbounded whole-repository rewrite.