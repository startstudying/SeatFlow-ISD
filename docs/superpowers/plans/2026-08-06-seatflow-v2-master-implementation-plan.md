# SeatFlow V2 Full Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild SeatFlow as a governed event-driven research platform for seat-stock capital diffusion and t+1/t+3/t+5 excess-return prediction.

**Architecture:** A standard monorepo contains a Python Research Engine, Spring Boot Control Plane, and Next.js Research Console. Iceberg/Parquet on MinIO is the research source of truth; PostgreSQL, Redis, Temporal, MLflow, Neo4j, OpenTelemetry, Docker Compose, and k3s provide governance, orchestration, tracking, projection, and deployment.

**Tech Stack:** Python 3.11+, PyTorch/PyG, PyArrow/Iceberg, FastAPI, gRPC, Temporal Python SDK; Java 21, Spring Boot, Flyway, Temporal Java SDK; Next.js 15, React 19, TypeScript; PostgreSQL, Redis, MinIO, MLflow, Neo4j, Docker Compose, Helm/k3s.

## Global Constraints

- Work directly on `main`, but create immutable tag `legacy-r4-before-isd-esd-rewrite` before destructive migration.
- `Cycle` is the atomic split unit; no cycle may cross train/validation/test/blind.
- Prediction is generated after t-day disclosure; earliest execution is t+1 open.
- Main labels are cumulative market-excess return and direction at t+1, t+3, and t+5.
- All features are point-in-time; only matured labels update seat-performance features.
- Base predictions used for Seat Residual must be rolling out-of-fold predictions.
- Entity IDs, ID hashes, and absolute-date integers never enter continuous feature MLPs.
- Future-impact graphs are sparse candidate graphs, not dense N×N adjacency matrices.
- Relation state and future-impact predictions must directly route HGT messages; final-head concatenation is not sufficient.
- Iceberg/Parquet is the training source of truth; Neo4j is only a query projection.
- A local 8GB GPU must support the complete workflow serially and resumably.
- Missing evidence or failed core gates produce `BLOCKED`, `INVALID`, or `DEGRADED`; never warning-only success.

---

## Program Map

| Phase | Scope | Depends on | Exit evidence |
|---|---|---|---|
| P0 | Safety baseline and monorepo | None | Recovery tag, legacy inventory, three application skeletons, common quality command |
| P1 | Domain schemas and contracts | P0 | JSON Schema, OpenAPI, AsyncAPI, Proto, deterministic generated clients |
| P2 | Infrastructure and observability | P0-P1 | Compose core/app profiles, PostgreSQL, MinIO/Iceberg, Temporal, trace propagation |
| P3 | Multi-source ingestion and Bronze | P1-P2 | Provider ports, adapters, raw lineage, revision/conflict handling, Bronze gates |
| P4 | PIT identity and Silver | P3 | PIT stock/seat/theme/event/tradeability facts and as-of query service |
| P5 | Event/Cycle/Sample/Universe | P4 | Cycle extension, daily samples, target/context pools, cycle-atomic walk-forward split |
| P6 | Features, labels, Base OOF, Residual | P5 | Feature registry, executable labels, mature seat features, OOF residual, Dataset Seal |
| P7 | Dynamic graph and model samples | P6 | Typed edges, sparse candidates, 20-day snapshots, out-of-core PyG samples, graph gates |
| P8 | Baselines and cost-aware evaluation | P7 | Tabular/sequence/graph baselines, unified metrics, executable backtest |
| P9 | SeatFlow-ESD core | P7-P8 | Event encoder, relation state, three future-graph decoders, state-routed HGT, Direct/Spillover |
| P10 | Training, 8GB mode, experiments, Blind | P8-P9 | Three-stage trainer, MemoryPlan, resume, Fold/Seed/ablation matrix, sealed Blind |
| P11 | Python API, gRPC, Temporal Activities | P6-P10 | Contract-first APIs and idempotent data/training/daily activities |
| P12 | Spring Boot control plane | P1-P2-P11 | RBAC, workflows, experiments, model registry, approvals, audit, resumable SSE |
| P13 | Next.js research console | P1-P12 | Governance, lifecycle, graph, experiment, prediction, backtest, paper-trading pages |
| P14 | Model deployment and daily operation | P10-P13 | Immutable model packages, paper broker, drift gates, governed daily workflow |
| P15 | CI, E2E, k3s, recovery, release | P0-P14 | Cross-language CI, Compose E2E, Helm/k3s, restore drill, evidence-based release gate |

## P0 - Safety Baseline and Monorepo

### P0-T1 Freeze legacy state

**Files:**
- Create `archive/legacy-r4/README.md`
- Create `archive/legacy-r4/legacy-status.json`
- Create `docs/migration/legacy-r4-inventory.md`

**Interfaces:** `legacy-status.json` records `source_git_sha`, tag, `model_valid=false`, data status, artifact hashes, and test summary.

- [ ] Record current SHA and working-tree status.
- [ ] Run current Python, Java, and frontend tests and preserve the real summary.
- [ ] Create tag `legacy-r4-before-isd-esd-rewrite`.
- [ ] Mark old checkpoints, data, backtests, and blind claims as `LEGACY_UNVERIFIED`.
- [ ] Verify the tag points to the pre-rewrite SHA.
- [ ] Commit `chore: freeze legacy r4 before v2 rewrite`.

### P0-T2 Scaffold target monorepo

Create `apps/web`, `apps/control-api`, `services/research-engine`, `packages/api-contracts`, `packages/domain-schemas`, `packages/generated-clients`, `packages/ui-kit`, `datasets/seatflow-isd`, `experiments`, `infra`, `tools`, `docs`, and `archive`. Add `.editorconfig`, root version metadata, root README, and Windows/Linux quality scripts.

### P0-T3 Initialize three applications

- Python: `services/research-engine/pyproject.toml`, `src/seatflow/api/main.py`, health test.
- Java: `apps/control-api/pom.xml`, Java 21 Spring Boot application, actuator test.
- Web: `apps/web/package.json`, App Router shell, health-state test.
- All unimplemented business pages must show explicit empty or blocked state.

### P0-T4 Common quality pipeline

Create `scripts/quality.sh`, `scripts/quality.ps1`, and `.github/workflows/v2-ci.yml`. Profiles: `layout`, `contracts`, `python`, `java`, `web`, `infra`, `full`. Any core failure returns non-zero; no `continue-on-error` for gates.

## P1 - Contracts and Code Generation

### P1-T1 Domain schemas

Create versioned schemas for Event, Cycle, Sample, Universe, GraphSnapshot, DatasetManifest, QualityReport, Prediction, WorkflowEvent, LogEvent, ModelPackage, and ReleaseEvidence. Positive and negative fixtures must prove required IDs, cutoff times, hashes, masks, and schema versions. Schema validation must reject IDs inside continuous feature lists.

### P1-T2 Research API

Create `packages/api-contracts/openapi/research-api.v1.yaml`. Large results return `artifactUri` and `manifestHash`; all errors include `code`, `message`, `traceId`, `details`, and `retryable`.

### P1-T3 Control API and SSE

Create `control-api.v1.yaml` and `workflow-events.v1.yaml`. All writes use `idempotencyKey`; SSE events have stable `eventId`, `workflowId`, monotonic `sequence`, and `occurredAt`; `Last-Event-ID` resumes delivery.

### P1-T4 gRPC

Create `seatflow.research.v1.InferenceService` with `BatchPredict`, `BatchScoreCandidates`, `ExplainPredictionBatch`, and `GetPropagationSubgraph`. Requests contain dataset/model versions and snapshot URIs, never complete DataFrames or PyG graphs.

### P1-T5 Deterministic code generation

Create cross-platform generation scripts. Generated Python, Java, and TypeScript clients include the source-contract hash. CI regenerates and requires `git diff --exit-code packages/generated-clients`.

## P2 - Infrastructure and Observability

### P2-T1 Compose profiles

`infra/compose/compose.yml` profiles:
- `core`: PostgreSQL, Redis, MinIO, Temporal, Temporal UI, lightweight OTel collector.
- `app`: web, control-api, research-api, Java worker, Python worker.
- `research`: MLflow.
- `graph`: Neo4j.
- `observability`: Prometheus, Grafana, Loki.

Core must run without MLflow, Neo4j, or dashboards. Formal training is blocked if required tracking is unavailable; smoke runs may use a clearly marked local tracker.

### P2-T2 PostgreSQL metadata schema

Flyway migrations create users, roles, permissions, dataset versions, build runs, experiment runs, model versions, workflow runs, approval records, audit logs, and idempotency records. Version tables store manifest hash and Git SHA. Secrets are references or encrypted values, never plaintext.

### P2-T3 MinIO and Iceberg

Create idempotent bucket/catalog initialization for `seatflow-data`, `seatflow-artifacts`, and `seatflow-mlflow`. Ensure five logical namespaces: bronze, silver, gold, graph, sample.

### P2-T4 Temporal queues

Queues: `seatflow-java-control`, `seatflow-python-data`, `seatflow-python-training`, `seatflow-python-inference`. Activities must heartbeat, retry only retryable errors, support cancellation, and exchange lightweight ArtifactRef payloads.

### P2-T5 Trace propagation

Propagate `X-Trace-Id` through browser, Java, Python, gRPC, Temporal, and structured logs. Logs include workflow/run/dataset/model identifiers where applicable.

## P3 - Multi-source Ingestion and Bronze

### P3-T1 Provider ports

Define ports for daily bars, LHB events, LHB seat details, themes, corporate actions, and tradeability. Every raw record requires source record ID, event time, available time, source version, and source confidence.

### P3-T2 Idempotent Bronze writer

Key records by source, source record ID, and revision. Same payload hash is a no-op; changed payload creates a new revision. Preserve raw payload hash and ingestion run ID.

### P3-T3 Adapters

Implement Tushare, exchange disclosure, Eastmoney, THS, authorized-file import, and AKShare adapters behind the same contract tests. Missing seat-detail permission returns `DEGRADED`/`PERMISSION_DENIED`; no adapter may infer fake seats.

### P3-T4 Conflict resolution

Keep all conflicting source evidence. Resolution decisions record selected/rejected sources, reason, and confidence. Unresolvable high-impact conflicts block downstream construction.

### P3-T5 Bronze gates

Check required timestamps, duplicate keys, hash collisions, schema drift, date coverage, and source capability. Any future-availability violation is a blocker.

## P4 - PIT Identity and Silver

### P4-T1 Stock identity

Build point-in-time stable stock identity, listing/ST/board state, delisting, and corporate-action intervals. Future names or classifications cannot backfill history.

### P4-T2 Seat identity

Build versioned canonical seat identity, aliases, seat type, confidence, and unknown-seat path. Similar strings cannot be auto-merged at high confidence without evidence. Identity changes create a new dataset version.

### P4-T3 Theme identity

Unify industry and concept entities while preserving `theme_type`, effective intervals, core membership, and confidence. Overlapping effective intervals are blockers.

### P4-T4 Event and participation facts

`event_id = event_date + stock_id + disclosure_batch`. Preserve buy and sell participation separately; do not net them into one seat-stock edge. Same-day multiple reasons normally use multi-hot reason unless the source proves separate disclosure batches.

### P4-T5 Tradeability and as-of query

Canonical statuses: `VALID_TRADABLE`, `SUSPENDED_AT_ENTRY`, `ONE_PRICE_LIMIT_UP`, `ONE_PRICE_LIMIT_DOWN`, `MISSING_ENTRY_PRICE`, `DELISTED_DURING_HORIZON`, `CORPORATE_ACTION_SPECIAL`. Research samples may be retained, but non-executable cases have `backtest_mask=0`.

## P5 - Lifecycle, Universe, and Splits

### P5-T1 Trading calendar

All offsets and distances use market trading days, including holidays and cross-year ranges. Individual-stock suspension does not change the market calendar.

### P5-T2 CycleBuilder

`CycleBuilder.build(stock_id, events, calendar) -> list[Cycle]`.
- First LHB creates a cycle ending at event +5 trading days.
- New event on or before current cycle end stays in the same cycle and resets end to new event +5.
- Event after cycle end creates a new cycle.
- Stable cycle ID derives from stock and cycle start.

### P5-T3 SampleBuilder

Create exactly one main sample per stock and prediction date for every active-cycle trading day. Store cutoff, days since last LHB, and `CURRENT_LHB`/`LHB_LAG_1...5`. Assert every input has `available_time <= cutoff_time`.

### P5-T4 UniverseBuilder

Target is the active-cycle pool. Context candidates come from shared seats, seat relay, coalition relay, industry, concept, and dynamic propagation, with default per-target limits 30/20/20/20/30/20. Context participates in message passing but not the main target loss unless selected for Spillover supervision.

### P5-T5 Walk-forward split

Use expanding training, approximately 60 trading days validation, at least 5-day embargo, approximately 20-day test, and final approximately six-month blind. Entire cycles go to one split. Train-only normalization and OOV handling are mandatory.

## P6 - Features, Labels, Base OOF, and Residual

### P6-T1 Feature Registry

Every feature declares entity type, source tables, lookback, availability rule, normalization, missing policy, and continuous/discrete status. Unregistered features cannot enter training. ID-like features are rejected.

### P6-T2 PIT features

Build 60-day stock sequence; 5/20/60/120-day seat windows; Event, Theme, and Market features. Base-market channels may use controlled denoising; event anomalies retain direction and are only robustly scaled. Seat outcome statistics use only matured labels.

### P6-T3 Labels

Use t+1 open as entry and t+1/t+3/t+5 close as exits. Store raw return, market-excess return, industry-excess auxiliary return, direction, daily path, maturity status, and tradeability masks. Daily path and cumulative labels must agree within numerical tolerance.

### P6-T4 Base Market

Train initial linear/LightGBM/CatBoost baselines on the eligible full-market universe without Seat/Event features. Produce three horizons under the same walk-forward splits.

### P6-T5 OOF and residual

Each Base prediction must come from a model that did not train on the target cycle. Save source fold, model hash, dataset hash. Residual generation is blocked if OOF coverage is incomplete.

### P6-T6 Dataset Seal

Manifest includes source, feature, label, cycle, and split policy hashes; row/node/edge/sample counts; all gate results. Critical failure sets dataset `INVALID` and `training_allowed=false`.

## P7 - Dynamic Graph and Samples

### P7-T1 Typed edge schema

Define Seat→Event buy/sell, Event→Stock impact, Stock↔Theme, Event→Event propagation, and Stock→Stock relay/diffusion edges with reverse relations, time rules, and transforms. Numeric identity features must be zero.

### P7-T2 Edge builders

Event→Stock persists for days-since-event 0..5. Event→Event only uses information available by the sample cutoff. Stock→Stock edges require economic evidence and confidence.

### P7-T3 Future candidates

Generate sparse candidate sets per horizon. t+1 emphasizes direct persistence, t+3 relay/diffusion, t+5 decay/reversal. Retain `candidate_reason`; apply per-target Top-K.

### P7-T4 Incremental snapshots

Store daily node state, edge additions/updates/expiry, stable ID mapping, target/context masks, market context, edge time delta, and schema version. Assemble `G[t-19:t]` on demand.

### P7-T5 Out-of-core PyG dataset

Read local target/context subgraphs per date or cycle; keep graph storage on CPU/disk and only current micro-batch on GPU. IDs become embedding indices; continuous transforms are fold-local.

### P7-T6 Graph gates

Check duplicate static edges, reverse relations, hard size limits, numeric identity fields, target index, label path, masks, and one-main-sample-per-day.

## P8 - Baselines, Evaluation, and Backtest

### P8-T1 Common artifact contract

All models produce the same return predictions, direction logits, rank scores, risk scores, optional graph outputs, masks, units, and hashes.

### P8-T2 Non-graph/sequence baselines

Implement at least one tabular baseline and one sequence baseline. They use the same splits and labels and cannot use propagation edges.

### P8-T3 Graph baselines

Implement static HGT/GAT, daily-event HGT, node-memory HGT, and explicit concat-SSM+HGT baseline. The concat baseline must not secretly route messages with state.

### P8-T4 Metrics

Return: MAE/Huber. Direction: accuracy/AUC. Ranking: IC, RankIC, ICIR, RankICIR, NDCG@K. Graph: Edge AUC/AUPR, Hit@K, Recall@K, direction accuracy. Group by Direct/Spillover, event type, horizon, and market regime.

### P8-T5 Cost-aware backtest

Implement t+1 open execution, T+1, suspension, one-price limits, delayed exits, fees, slippage, impact cost, and Top-K portfolios. Report gross/net identity, turnover, cost attribution, Sharpe, drawdown, and executable coverage.

## P9 - SeatFlow-ESD Core

### P9-T1 Event encoder

Encode Seat, Event, Stock, Theme, and Market context. Market is shared context, not a fully connected supernode. Produce node embeddings and relation observations `O_t^r`.

### P9-T2 Relation state

`RelationStateCell.forward(previous_state, observation, delta_trade_days, market_context, relation_type)` implements historical decay, gated event injection, same-direction strengthening, direction reversal, and sparse storage.

### P9-T3 Future graph decoders

Three horizon-specific decoders predict existence, direction, and strength only for candidate edges. Add independent labels for seat re-entry, cross-stock migration, coalition recurrence, related-stock activation, and direction.

### P9-T4 State-routed HGT

Messages use predicted future-edge probability/direction/strength inside filtering, message transformation, and attention. Tests must prove state changes model output and routing differs from final-head concatenation.

### P9-T5 Direct/Spillover and losses

Direct supervises event stocks; Spillover supervises linked non-event stocks. Joint loss combines return, same-date ranking, direction, graph, and state constraints. Empty masks must remain numerically stable.

### P9-T6 Explanation

Extract actual activated paths such as Seat→Event→Stock→Theme→Stock, relation contributions, persistence, coalition, crowding, risk, and horizon differences. Do not present attention as causality.

## P10 - Training, 8GB, Experiments, Blind

### P10-T1 Three-stage trainer

Stage A event graph/state, Stage B future graph, Stage C joint multi-task. Each stage has its own checkpoint and validation-only early stopping.

### P10-T2 MemoryPlan

Profiles: `local-smoke`, `local-full-slow`, `server-full`. `local-full-slow` retains full history, 20-day graph, 60-day stock sequence, all folds/seeds/ablations/blind; it only reduces micro-batch, hidden size, neighbors, workers, and concurrency. Never silently switch to CPU.

### P10-T3 Resume and GPU lifecycle

Checkpoint model, optimizer, scheduler, scaler, RNG, effective config, hashes, and progress. Hash mismatch blocks resume. At Fold/Seed boundaries move state to CPU and release GPU references.

### P10-T4 Experiment matrix

Register all required baselines and ablations: no relation state, no future graph, concat instead of routing, no event node, no spillover, same decoder, standard denoising, no graph supervision. Each RunSpec has a stable hash and immutable artifact namespace.

### P10-T5 Walk-forward/multi-seed

Run each Fold and Seed serially on local hardware, persist after every Fold, resume safely, and report distributions rather than only the best Seed.

### P10-T6 Blind

Seal dataset manifest, model checkpoint, config, code SHA, and candidate hash. Blind labels remain inaccessible during development. Evaluation requires approval token, is audited, and produces non-overwritable results.

## P11 - Python Services and Activities

Implement contract-first Research REST routes, gRPC inference, dataset-build Activities, training/evaluation Activities, and daily inference/drift Activities. Activities exchange ArtifactRef only, heartbeat partition/epoch progress, support cancellation, and fail closed on non-retryable data/research gates.

## P12 - Java Control Plane

Package by feature: security, dataset, workflow, experiment, model, approval, audit, notification. Java may orchestrate and store metadata but may not implement Cycle, labels, graph candidates, losses, metrics, or return calculations.

Required state machines:
- Workflow: `REQUESTED -> RUNNING -> BLOCKED|FAILED|COMPLETED`.
- Model: `DRAFT -> VALIDATED -> BLIND_APPROVED -> DEPLOYED -> RETIRED`.

All writes are idempotent and audited. SSE persists events before delivery and supports Last-Event-ID recovery.

## P13 - Next.js Research Console

Deliver application shell/UI kit and pages for overview, data sources, dataset builds, quality, Event/Cycle/Sample, seat identity review, feature registry, graph snapshots, propagation graph, experiment creation/monitoring/comparison, future graph evaluation, Direct/Spillover predictions, stock explanation, backtest/cost, Blind approval, model registry, daily inference, paper trading, settings, RBAC, audit, and monitoring.

Every page supports loading, empty, error, degraded, blocked, and permission states. Fixture data is visually and semantically marked and never shown as formal result.

## P14 - Deployment, Paper Trading, Drift, Daily Workflow

Build immutable model packages containing model state, effective config, feature/edge/label hashes, and Git SHA. Add a paper broker/risk engine with accounting invariants and explicit rejection reasons; no real brokerage calls. Add data/model/graph drift gates. Daily workflow executes ingest→build→predict→explain→drift→paper signal, is idempotent by date/model, and blocks on any gate.

## P15 - CI, E2E, k3s, Recovery, Release

- CI order: contracts→Python→Java→Web→Infra→E2E.
- Compose E2E uses deterministic fixture data and verifies both failure and success paths.
- Helm/k3s uses the same images and environment semantics as Compose; GPU workers are isolated by node selector and resource quota.
- Backup/restore covers PostgreSQL, versioned MinIO data, model packages, and audit evidence; Redis is not a source of truth.
- Release verification requires Dataset Gate, Walk-Forward, multi-seed, baselines/ablations, cost-aware backtest, Blind approval, model package, E2E, image digest, schema compatibility, and migration evidence. Missing evidence produces `BLOCKED`.

## Completion Definition

1. P0-P15 each has a completion report and real commit SHA.
2. Contracts and generated clients have no drift.
3. Dataset Manifest passes PIT, maturity, ID, Cycle, label, graph, and coverage gates.
4. Base OOF and Seat Residual have independent lineage.
5. Relation state, future graph, and state routing have unit, supervision, and ablation evidence.
6. All baselines and the full model share split, label, cost, and evaluator.
7. Walk-forward, multi-seed, ablation, Direct/Spillover, and net backtest reports are complete.
8. Blind candidate was sealed and approved; Blind results are immutable and not used for tuning.
9. Compose E2E, Helm/k3s, backup/restore, and release gates pass.
10. The UI reports evidence states honestly and never substitutes fixtures for formal results.
