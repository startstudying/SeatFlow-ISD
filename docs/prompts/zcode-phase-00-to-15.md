# SeatFlow V3｜ZCode Phase 00–15 阶段提示词

> 前置条件：ZCode 已加载 `docs/prompts/zcode-master-prompt.md`。每次只复制一个 Phase。完成后必须停止并输出执行报告，不得自动进入下一 Phase。

---

## Phase 00｜安全边界、契约、骨架与 CI

```text
执行 SeatFlow V3 Phase 00。

目标：
建立安全重写边界、三个应用骨架、跨语言契约、基础设施配置和 CI。此阶段不实现数据业务、模型、正式预测或正式回测。

实施计划任务：
Task 1、Task 2、Task 3、Task 4。

必须完成：
1. 验证当前分支为 rewrite/seatflow-v3。
2. 创建并推送 legacy-before-seatflow-v3 Tag；若已存在，验证它指向 9d7cda4a2bce1a11f6e125475de938666cdd7140，不得移动。
3. 创建 research-engine、control-center、research-console、contracts、infra 和 legacy-reference 骨架。
4. 冻结 Seat、Stock、LHBEvent、SeatParticipation、EventCycle、PredictionSample、PropagationCandidate、TaskStatus、SeatFlowError、ArtifactReference JSON Schema。
5. 冻结 Control API、Research API 和 Task SSE 契约。
6. 建立 PostgreSQL、Redis、MinIO、MLflow、Prometheus、Grafana Compose 配置。
7. 建立 Python、Java、Frontend CI。
8. 三个应用只实现健康检查和明确的“无正式模型”空态。

测试：
- pytest contracts/tests -v
- pytest infra/tests -v
- pytest research-engine/tests/test_package_boundary.py -v
- mvn -f control-center/pom.xml test
- npm --prefix research-console run lint
- npm --prefix research-console run typecheck
- npm --prefix research-console test
- npm --prefix research-console run build
- docker compose -f infra/compose/docker-compose.yml config

禁止：
- 迁移旧 DT-SeatHGT。
- 生成假数据指标或预测。
- 直接修改 main。
- 引入 Temporal、Iceberg 或 Neo4j 硬依赖。

阶段验收：
- 三个应用可独立启动。
- 契约测试全绿。
- Compose 配置合法。
- CI 文件存在且命令与本地一致。
- 页面明确显示 No formal model is active。

完成后停止，输出提交 SHA 和全部测试结果。
```

---

## Phase 01｜Bronze 数据采集、血缘与多源冲突

```text
执行 SeatFlow V3 Phase 01。

目标：
实现不可变 Bronze 记录、数据源接口、Tushare/AKShare/授权文件适配器、能力探测、Manifest、Parquet 分区写入和多源冲突报告。

实施计划任务：
Task 5、Task 6。

必须完成：
1. timezone-aware event_time、available_time、ingested_at。
2. source、source_record_id、source_version、revision_no、raw_payload_hash、ingestion_run_id、source_confidence。
3. canonical JSON + SHA-256 Manifest。
4. provider transport 注入，测试禁止真实网络。
5. 权限不足返回 DEGRADED 能力和机器可读原因。
6. 冲突记录保留全部 Bronze 事实，Silver 选择规则有版本和证据。
7. 适配器覆盖 stock master、calendar、daily bar、daily basic、LHB summary、LHB participation、industry、concept、tradeability。

测试：
- pytest research-engine/tests/ingestion -v
- ruff check research-engine/src/seatflow/ingestion research-engine/tests/ingestion
- mypy research-engine/src/seatflow/ingestion

禁止：
- 在模块 import 时读取真实 Token。
- 跨来源覆盖原始记录。
- 席位明细缺失时用汇总金额伪造席位。

阶段验收：
- 同一输入重复执行生成同一 Manifest Hash。
- 冲突报告可追溯来源、规则和置信度。
- 所有记录保留 available_time。

完成后停止。
```

---

## Phase 02｜股票与席位身份 Point-in-Time 归一

```text
执行 SeatFlow V3 Phase 02。

目标：
实现股票稳定身份、席位别名、席位类型、历史有效区间、置信度和冲突处理。

实施计划任务：
Task 7。

必须完成：
1. resolve_stock(raw_code, at_time)。
2. resolve_seat(raw_name, at_time)。
3. 历史别名和 valid_from/valid_to。
4. BROKER_BRANCH、INSTITUTION、NORTHBOUND_SH、NORTHBOUND_SZ、SPECIAL_SEAT、UNKNOWN。
5. 无法识别席位生成 provider-scoped 稳定 UNKNOWN ID。
6. UNKNOWN ID 只能用于离散映射，不得进入连续特征。
7. 重叠区间和别名冲突必须显式失败或进入冲突状态。

测试：
- pytest research-engine/tests/identity -v
- ruff check research-engine/src/seatflow/identity research-engine/tests/identity
- mypy research-engine/src/seatflow/identity

阶段验收：
- 当前别名不会覆盖历史身份。
- 同一原始输入和时点解析结果稳定。
- 每次解析返回 confidence 和 evidence。

完成后停止。
```

---

## Phase 03｜Event、Cycle 与三类 Universe

```text
执行 SeatFlow V3 Phase 03。

目标：
实现完整 LHBEvent、SeatParticipation、EventCycle、Direct Pool、Spillover Pool、Base Pool 和 Propagation Candidate。

实施计划任务：
Task 8、Task 9。

必须完成：
1. Event 按 stock + trade_date + disclosure_batch 聚合。
2. 所有买卖席位保留独立 participation 行。
3. Cycle 使用交易日历：latest event + 5 trading days。
4. 周期内重复上榜延长同一 Cycle；周期结束后创建新 Cycle。
5. Direct Pool：0 <= days_since_last_event <= 5。
6. Spillover Pool 排除同日直接上榜股票。
7. Candidate 支持 same-seat、same-coalition、industry、concept、capital-rotation 和辅助 comovement。
8. 每种关系独立 Top-K，排序和 tie-break 确定性。

测试：
- pytest research-engine/tests/dataset/test_events.py -v
- pytest research-engine/tests/dataset/test_cycles.py -v
- pytest research-engine/tests/dataset/test_universe.py -v
- pytest research-engine/tests/dataset/test_propagation_candidates.py -v

阶段验收：
- Cycle 延长边界测试全绿。
- Direct 与 Spillover 无同日目标重叠。
- Candidate 只使用 cutoff 前证据。

完成后停止。
```

---

## Phase 04｜PIT 特征、标签、Sample、Split 与 Dataset Seal

```text
执行 SeatFlow V3 Phase 04。

目标：
完成特征注册表、Stock/Seat/Event/Theme/Market 特征、五日路径、t+1/t+3/t+5 标签、可交易状态、滚动 Sample、Cycle 原子切分、质量门禁、Manifest 和 Seal。

实施计划任务：
Task 10、Task 11、Task 12。

必须完成：
1. 每个特征声明来源、lookback、availability、maturity、transform、normalization、missing、dtype、kind 和 lineage。
2. Stock 60 日，Seat 5/20/60/120 日，Graph context 20 日。
3. 事件异常保留突变，只做 log1p、fold-fitted robust scaling 和 fold-fitted clipping。
4. 标签使用 t+1 Open 入场，t+1/t+3/t+5 Close 退出。
5. 日级路径与累计标签严格一致。
6. Event-study mask 与 backtest mask 分离。
7. 同一 Cycle 只能进入一个 split。
8. Dataset Seal 对 PIT、ID leakage、maturity、Cycle split、label consistency、PK 和 graph reference fail-closed。

测试：
- pytest research-engine/tests/features -v
- pytest research-engine/tests/labels -v
- pytest research-engine/tests/dataset -v

硬断言：
- available_time violations = 0
- numeric identity continuous features = 0
- immature label profile rows = 0
- cycle cross-split = 0
- path compounding errors = 0

阶段验收：
- 合成合法数据可生成 Seal。
- 注入任一违规后 Seal 被拒绝。

完成后停止。
```

---

## Phase 05｜图 Schema、快照与张量化

```text
执行 SeatFlow V3 Phase 05。

目标：
实现 Seat/Event/Stock/Theme 动态异构图 Schema、快照、稳定局部索引、Mask、Top-K 和 PyG HeteroData 张量化。

实施计划任务：
Task 13。

必须完成：
1. 节点 Seat、Event、Stock、Theme。
2. Market 作为 graph-level context。
3. 关系包含 seat_buy_event、seat_sell_event、event_impacts_stock、stock_belongs_theme、same_stock_repeat、same_seat_relay、same_coalition_relay、same_theme_diffusion、capital_rotation、spillover_candidate。
4. target_mask、context_mask、loss_mask、backtest_mask 分离。
5. 禁止数值 ID、ID Hash、绝对日期整数和未来信息进入边连续特征。
6. 快照可序列化、校验 Hash、加载并产生相同张量。
7. 默认最多两跳传播。

测试：
- pytest research-engine/tests/graph -v

阶段验收：
- round-trip 后节点、边、特征和关系计数一致。
- 悬空节点引用被拒绝。
- Top-K 结果确定性。

完成后停止。
```

---

## Phase 06｜Market Base Model 与严格 OOF

```text
执行 SeatFlow V3 Phase 06。

目标：
实现不读取席位图信息的 Market Base Encoder、五日路径输出、Market Regime、Fold Trainer 和严格 Walk-Forward OOF。

实施计划任务：
Task 14。

必须完成：
1. Base 输入只含 stock sequence 和 market context。
2. 输出 base_path[5]、R1/R3/R5 和 regime embedding。
3. R1/R3/R5 从 path 累计得到。
4. Fold 内拟合 scaler、clip 和 calibration。
5. OOF 每行记录 fitted cycle set 或可审计训练边界。
6. 8GB 配置支持 AMP、gradient accumulation 和 checkpoint resume。

测试：
- pytest research-engine/tests/models/test_base_shapes.py -v
- pytest research-engine/tests/training/test_oof_independence.py -v
- pytest research-engine/tests/training/test_base_micro_overfit.py -v

阶段验收：
- Base API 无 Seat/Event/Graph 输入。
- OOF independence 测试全绿。
- 微型确定性数据 loss 至少下降 95%。

完成后停止。
```

---

## Phase 07｜Seat Memory、Event Encoder 与 Direct

```text
执行 SeatFlow V3 Phase 07。

目标：
实现席位行为记忆、事件交互编码、Direct Effect Decoder 和 Direct 基线矩阵。

实施计划任务：
Task 15。

必须完成：
1. Seat Memory 使用 5/20/60/120 窗口和成熟标签掩码。
2. Event Encoder 保留全部 participation，建模协同、冲突、集中度、支配度、原因、新颖性和延续。
3. Direct 输出 path[5]、R1/R3/R5、continuation、reversal、uncertainty。
4. 训练主目标使用 Base OOF Residual。
5. 支持 Amount Only、Seat Type、Anonymous Seat、Real Seat、Shuffled Seat。
6. 所有基线共享数据、split 和 trainer，仅通过配置改变信息。

测试：
- pytest research-engine/tests/models/test_seat_memory.py -v
- pytest research-engine/tests/models/test_event_encoder.py -v
- pytest research-engine/tests/models/test_direct_model.py -v
- pytest research-engine/tests/training/test_direct_micro_overfit.py -v

阶段验收：
- 未成熟历史变化不影响 Seat embedding。
- participation 行重排不改变事件聚合语义。
- Direct 微型过拟合成功。

完成后停止。
```

---

## Phase 08｜Relation State 与 Spillover

```text
执行 SeatFlow V3 Phase 08。

目标：
实现稀疏 Relation State Gate、门控异构消息传递、Top-K 激活和 Spillover Decoder。

实施计划任务：
Task 16。

必须完成：
1. Relation State 输出 activation、strength、direction、reliability、age、decay、regime sensitivity。
2. activation=0 时边消息严格为 0。
3. Relation State 进入邻居选择、消息和注意力，不允许只在预测头拼接。
4. Spillover loss 只作用于 Spillover Pool。
5. Direct 股票同日不得作为 Spillover target。
6. 输出 spillover path[5]、R1/R3/R5、probability、uncertainty。

测试：
- pytest research-engine/tests/models/test_relation_state.py -v
- pytest research-engine/tests/models/test_gated_message_passing.py -v
- pytest research-engine/tests/models/test_spillover_model.py -v

阶段验收：
- gate routing test 证明关系状态实际改变传播。
- Top-K 每关系独立生效。
- Spillover Mask 与 Direct Mask 隔离。

完成后停止。
```

---

## Phase 09｜SeatFlow-FDG 联合模型与训练

```text
执行 SeatFlow V3 Phase 09。

目标：
组装 Base、Seat/Event、Relation State、Direct、Spillover 和 Multi-Horizon Fusion，完成联合损失和训练器。

实施计划任务：
Task 17。

必须完成：
1. final_path = base_path + direct_path + spillover_path。
2. R1/R3/R5 从 final_path 累计。
3. Return、Rank、Direction、Relation、Spillover、Calibration、Risk Loss。
4. Rank 仅在同一 prediction_date 内比较。
5. Direct、Spillover、event-study、backtest Mask 隔离。
6. 所有模块获得 finite gradient。
7. Checkpoint 记录 Dataset Manifest Hash、Git SHA、Config Hash、Fold、Seed 和 memory plan。

测试：
- pytest research-engine/tests/models/test_fdg_output.py -v
- pytest research-engine/tests/models/test_fdg_gradients.py -v
- pytest research-engine/tests/training/test_loss_masks.py -v
- pytest research-engine/tests/training/test_fdg_micro_overfit.py -v

阶段验收：
- additive decomposition 精确通过。
- 所有核心模块梯度 finite。
- 全模型微型过拟合成功。

完成后停止。
```

---

## Phase 10｜Walk-Forward、消融、Placebo、校准与 Blind

```text
执行 SeatFlow V3 Phase 10。

目标：
实现完整研究证据链：指标、Walk-Forward、Embargo、多 Seed、消融、Placebo、校准、Blind Seal 和 Model Gate。

实施计划任务：
Task 18、Task 19。

必须完成：
1. MAE、Huber、AUC、IC、RankIC、ICIR、RankICIR、NDCG@K、Hit@K。
2. Edge AUC、AUPR、Recall@K、direction 和 calibration。
3. Brier、ECE、Reliability Curve、Uncertainty Coverage。
4. Base、Amount Only、Seat Type、Anonymous Seat、Real Seat、No Graph、Shuffled Seat、Shuffled Graph、Real Graph、Full FDG、No Routing 消融。
5. Placebo 保持样本量和约束，并由 seed 确定。
6. Blind 开发访问被拒绝。
7. Blind Candidate 冻结 dataset/code/config/checkpoint/cost/range。
8. Model Gate fail-closed。

测试：
- pytest research-engine/tests/evaluation -v

阶段验收：
- 指标参考值测试全绿。
- Cycle 原子、时间顺序和 Embargo 全绿。
- Blind 报告不可覆盖。
- 任一关键门禁失败会阻止 Promotion 和 Paper Trading。

完成后停止。
```

---

## Phase 11｜成本后回测与模拟盘内核

```text
执行 SeatFlow V3 Phase 11。

目标：
实现真实交易约束、成本、Top-K 组合、持有期重叠、订单生命周期、报告和 PaperAccount。

实施计划任务：
Task 20。

必须完成：
1. 佣金最低收费、卖出印花税、滑点和冲击成本。
2. t+1 Open 入场。
3. 停牌、一字涨停无法买、一字跌停无法卖、退市和特殊处理。
4. 订单失败、挂起、部分成交、撤单和审计原因。
5. 连续日期 Top-K 信号的现金和持仓约束。
6. 成本前、成本后、Direct、Spillover 和全组合报告。
7. PaperAccount 只接受 promoted 且 paper_trading_allowed 的模型。

测试：
- pytest research-engine/tests/backtest -v

阶段验收：
- 固定订单成本精确匹配。
- 不可执行订单不会虚构成交。
- 无效模型无法创建模拟盘会话。

完成后停止。
```

---

## Phase 12｜Python Research API、Worker 与 Artifact

```text
执行 SeatFlow V3 Phase 12。

目标：
实现 FastAPI Research API、Redis Queue、Worker、任务状态、取消、幂等、进度事件和 MinIO Artifact。

实施计划任务：
Task 21。

必须完成：
1. /research/v1/health。
2. 数据集、训练、评估、回测、推理任务 API。
3. 同一 idempotency key 返回同一 taskId。
4. Queue payload 只含 ID、Version、URI、Hash 和小参数。
5. PostgreSQL 或 Control callback 保存权威状态。
6. Redis 只负责队列、锁、缓存和进度。
7. Artifact 返回 URI、SHA-256、size、mediaType、schemaVersion。
8. 支持取消和 worker heartbeat。

测试：
- pytest research-engine/tests/api -v
- pytest research-engine/tests/tasks -v

阶段验收：
- OpenAPI 契约兼容。
- 重复提交不重复入队。
- Hash 不匹配时任务失败并不可重试。

完成后停止。
```

---

## Phase 13｜Spring Boot Control Center

```text
执行 SeatFlow V3 Phase 13。

目标：
实现 Control Center 的任务、用户权限、数据集、实验、模型、审批、回测、模拟盘、审计、SSE 和系统状态。

实施计划任务：
Task 22、Task 23、Task 24。

必须完成：
1. PostgreSQL + Flyway。
2. Task 状态机、乐观锁、幂等键、取消和审计。
3. Dataset Version、Quality Gate、Experiment、Run、Model Version、Promotion、Prediction、Backtest、Paper Session。
4. Dataset invalid 阻止 formal experiment。
5. Model invalid 阻止 Promotion。
6. 未 Promotion 或不允许 paper trading 阻止 Paper Session。
7. 独立权限：dataset seal、blind access、model promotion、paper activation。
8. SSE 支持 Last-Event-ID 重放。
9. 错误模型匹配 contracts。

测试：
- mvn -f control-center/pom.xml test
- mvn -f control-center/pom.xml verify

阶段验收：
- 非法状态转换被拒绝。
- RBAC、审批、审计和 SSE 重连测试全绿。
- Java 未复制 Python 金融语义。

完成后停止。
```

---

## Phase 14｜Next.js Research Console

```text
执行 SeatFlow V3 Phase 14。

目标：
实现完整研究控制台和端到端真实性状态。

实施计划任务：
Task 25、Task 26、Task 27、Task 28。

必须页面：
1. 总览。
2. 数据源。
3. Dataset Builder。
4. 数据质量。
5. Event/Cycle/Sample。
6. 席位身份与画像。
7. 传播图。
8. 实验创建和训练监控。
9. 模型比较。
10. 预测和单股票解释。
11. 回测和成本归因。
12. 模拟盘。
13. 模型注册与审批。
14. 用户、权限、审计和系统监控。

必须状态：
- Loading
- Empty
- Error
- Degraded
- Synthetic Non-Formal
- Invalid Model
- Formal Validation
- Formal Blind
- Promoted Model

必须显示：
- source
- cutoffTime
- datasetVersion
- modelVersion
- formal/blind
- costIncluded
- tradeability
- validity reason
- traceId

测试：
- npm --prefix research-console run lint
- npm --prefix research-console run typecheck
- npm --prefix research-console test
- npm --prefix research-console run build

阶段验收：
- 刷新后恢复任务状态。
- SSE 重放不重复指标。
- Invalid 模型不显示绿色正式预测。
- 传播路径显示 horizon、relation、probability、direction、strength、evidence time。

完成后停止。
```

---

## Phase 15｜部署、E2E、发布门禁、Blind 与每日推理

```text
执行 SeatFlow V3 Phase 15。

目标：
完成镜像、Compose/k3s、可观测性、合成 E2E、Runbook、正式协议冻结、Blind、Promotion 和每日推理入口。

实施计划任务：
Task 29、Task 30、Task 31、Task 32。

必须完成：
1. 三个非 root 应用镜像和 Health Check。
2. Compose 和 k3s 使用相同镜像、配置键和 Secret 规则。
3. Trace 关联 taskId、datasetVersion、runId、modelVersion、predictionDate。
4. 监控 queue、task duration、data latency、quality、GPU、fold/seed、API、SSE。
5. 合成 E2E：ingest -> identity -> cycle -> feature -> label -> graph -> sample -> seal -> base OOF -> direct -> spillover -> evaluation -> prediction -> Java -> UI。
6. 合成结果标记 SYNTHETIC_NON_FORMAL。
7. Release verifier 检查 tests、Seal、Git SHA、Model Gate、cost report 和 clean worktree。
8. 冻结 formal dataset/model/training/evaluation/blind/cost config。
9. Blind 只执行一次并写不可覆盖报告。
10. Promotion 仅在全部门禁通过时执行。
11. 每日推理记录 cutoff、Dataset、Model、graph hash、output URI、explanation URI。

测试：
- docker compose -f infra/compose/docker-compose.yml config
- pytest infra/tests -v
- pytest tools/release/tests -v
- python tools/e2e/run_synthetic_e2e.py
- npm --prefix research-console run test:e2e
- 全量 Python、Java、Frontend、契约测试

正式研究注意：
- 代码开发完成不等于模型有效。
- 如果正式实验为负，记录 INVALID_RESULT 和完整证据，不得伪造 Promotion。
- 不得自动使用真实券商账户或执行真实下单。

阶段验收：
- Synthetic E2E 全绿。
- Release verifier 可阻止缺证据发布。
- Blind 和 Promotion 状态不可绕过。
- 每日推理和模拟盘只接受 Promoted Model。

完成后停止并输出最终交付清单、测试矩阵、镜像、部署命令、正式证据状态和所有提交 SHA。
```

---

## 统一阶段报告模板

```text
SeatFlow V3 Phase Report

Phase：
实施计划 Task：
分支：
起始 SHA：
结束 SHA：

完成内容：
1.
2.
3.

RED 证据：
- 测试：
- 失败原因：

GREEN 证据：
- 命令：
- 结果：

静态检查：
- Python：
- Java：
- Frontend：
- Contracts：
- Compose：

文件变化：
- Create：
- Modify：
- Delete：

契约变化：
数据状态：
模型状态：
正式性声明：
已知限制：
下一 Phase 前置条件：
```

任何阶段报告缺少失败测试证据、通过测试命令或提交 SHA，均视为阶段未完成。