# SeatFlow V3 测试矩阵

- 状态：冻结测试基线
- 日期：2026-08-06
- 适用分支：`rewrite/seatflow-v3`
- 原则：测试不仅证明代码能运行，还必须证明数据不泄漏、模型机械结构正确、研究证据可审计、平台无法绕过门禁。

---

## 1. 测试层级

| 层级 | 目的 | 失败影响 |
|---|---|---|
| L0 Contract | JSON Schema、OpenAPI、事件和错误模型 | 阻止任何跨服务联调 |
| L1 Unit | 单函数、领域规则、模型部件 | 阻止对应 Task 提交 |
| L2 Property | 时间、切分、Hash、排序和数值不变量 | 阻止 Dataset Seal 或模型训练 |
| L3 Integration | 数据库、Redis、MinIO、API、Worker | 阻止应用集成 |
| L4 Model Mechanics | Shape、Mask、Gradient、Micro-overfit、路由 | 阻止正式实验 |
| L5 Research Protocol | Walk-Forward、OOF、Placebo、Blind、成本 | 阻止模型有效性结论 |
| L6 E2E | 三应用和完整合成链路 | 阻止发布候选 |
| L7 Formal Evidence | 正式 Dataset、Blind 和 Gate | 决定 Promotion 与模拟盘 |

---

## 2. 契约测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| CT-001 | 所有 JSON Schema 符合 Draft 2020-12 | `contracts/tests/test_json_schemas.py` | L0 | 是 |
| CT-002 | Schema 默认 `additionalProperties=false` | 同上 | L0 | 是 |
| CT-003 | 所有 `*Id` 字段为 string | 同上 | L0 | 是 |
| CT-004 | 金融时间记录包含 `availableTime` | 同上 | L0 | 是 |
| CT-005 | OpenAPI 可解析且每个 operation 有 `operationId` | `contracts/tests/test_openapi_contracts.py` | L0 | 是 |
| CT-006 | 所有 API 错误引用统一 `SeatFlowError` | 同上 | L0 | 是 |
| CT-007 | 长任务只返回 `taskId`，不内联大产物 | 同上 | L0 | 是 |
| CT-008 | ArtifactReference 含 URI、Hash、Size、MediaType、SchemaVersion | 同上 | L0 | 是 |
| CT-009 | SSE 事件匹配 task event schema | 同上 | L0 | 是 |
| CT-010 | 生成客户端与 OpenAPI Hash 一致 | `contracts/tests/test_generated_clients.py` | L0 | 是 |

---

## 3. 仓库边界和构建测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| BD-001 | V3 Python 不导入旧 `src/seatflow` | `research-engine/tests/test_package_boundary.py` | L1 | 是 |
| BD-002 | Java 不依赖旧 `backend-java` 包 | `control-center/src/test/.../ArchitectureTest.java` | L1 | 是 |
| BD-003 | Frontend 不读取 Python 私有路径 | `research-console/tests/architecture.test.ts` | L1 | 是 |
| BD-004 | Git 忽略 `.env`、数据、Checkpoint、数据库目录 | `tools/release/tests/test_repo_hygiene.py` | L1 | 是 |
| BD-005 | Compose 定义全部核心服务和 Health Check | `infra/tests/test_compose_config.py` | L1 | 是 |
| BD-006 | Compose 不包含真实凭证 | 同上 | L1 | 是 |
| BD-007 | 三个应用镜像非 root | `infra/tests/test_images_and_manifests.py` | L1 | 是 |
| BD-008 | Compose 和 k3s 镜像、配置键一致 | 同上 | L1 | 是 |

---

## 4. Bronze 与多源数据测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| DA-001 | BronzeRecord 必填时间、来源、版本和 Hash | `tests/ingestion/test_bronze_records.py` | L1 | 是 |
| DA-002 | 时间必须 timezone-aware | 同上 | L1 | 是 |
| DA-003 | canonical Manifest Hash 确定性 | `tests/ingestion/test_manifest.py` | L2 | 是 |
| DA-004 | 相同输入重复写入不产生不同 Manifest | 同上 | L2 | 是 |
| DA-005 | 原始记录不可跨来源覆盖 | `tests/ingestion/test_reconcile.py` | L1 | 是 |
| DA-006 | 冲突报告保留全部源值和规则版本 | 同上 | L1 | 是 |
| DA-007 | 权限不足返回 DEGRADED，不伪造完整数据 | `tests/ingestion/test_provider_contract.py` | L1 | 是 |
| DA-008 | 适配器返回 availability evidence | 同上 | L1 | 是 |
| DA-009 | 席位明细缺失不由汇总金额推导假席位 | 同上 | L1 | 是 |
| DA-010 | Parquet 分区写入和读取行数、Hash 一致 | `tests/ingestion/test_parquet_writer.py` | L2 | 是 |

---

## 5. 身份归一测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| ID-001 | 股票代码在指定时点解析为稳定 ID | `tests/identity/test_stock_identity.py` | L1 | 是 |
| ID-002 | 席位旧别名按历史有效区间解析 | `tests/identity/test_seat_identity.py` | L1 | 是 |
| ID-003 | 当前别名不覆盖历史身份 | 同上 | L2 | 是 |
| ID-004 | 重叠有效区间冲突被拒绝 | 同上 | L1 | 是 |
| ID-005 | UNKNOWN 席位 ID 稳定且 provider-scoped | 同上 | L2 | 是 |
| ID-006 | 身份解析返回 confidence 和 evidence | 同上 | L1 | 是 |
| ID-007 | 不同真实席位不会因文本规范化碰撞 | 同上 | L2 | 是 |
| ID-008 | 数值 ID/Hash 不进入连续特征 | `tests/features/test_identity_leakage.py` | L2 | 是 |

---

## 6. Event、Cycle、Universe 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| EC-001 | 同股票、日期、批次形成唯一 Event | `tests/dataset/test_events.py` | L1 | 是 |
| EC-002 | 买卖席位 participation 全部保留 | 同上 | L1 | 是 |
| EC-003 | Event ID 构造确定性 | 同上 | L2 | 是 |
| EC-004 | 首次事件 cycle_end=事件后第 5 交易日 | `tests/dataset/test_cycles.py` | L1 | 是 |
| EC-005 | 周期内重复事件延长 Cycle | 同上 | L1 | 是 |
| EC-006 | 周期结束后事件创建新 Cycle | 同上 | L1 | 是 |
| EC-007 | Calendar 使用交易日而非自然日 | 同上 | L2 | 是 |
| UN-001 | Direct Pool 仅含活跃 Cycle 股票 | `tests/dataset/test_universe.py` | L1 | 是 |
| UN-002 | 同日直接上榜股票不进入 Spillover Pool | 同上 | L1 | 是 |
| UN-003 | Context Stock 不参与 Direct Loss | 同上 | L2 | 是 |
| UN-004 | Base Pool 不读取当前 Seat/Event 信息 | 同上 | L2 | 是 |
| UN-005 | Candidate 只使用 cutoff 前证据 | `tests/dataset/test_propagation_candidates.py` | L2 | 是 |
| UN-006 | 每关系 Top-K 和 tie-break 确定性 | 同上 | L2 | 是 |
| UN-007 | 默认传播路径不超过两跳 | 同上 | L2 | 是 |

---

## 7. 特征和 Point-in-Time 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| FT-001 | 每个特征注册字段完整 | `tests/features/test_registry.py` | L1 | 是 |
| FT-002 | 所有输入 `available_time <= cutoff` | `tests/features/test_pit_features.py` | L2 | 是 |
| FT-003 | Seat 收益画像只含成熟标签 | 同上 | L2 | 是 |
| FT-004 | Scaler 仅在当前训练 Fold 拟合 | `tests/features/test_scaling.py` | L2 | 是 |
| FT-005 | Clip 阈值仅在当前训练 Fold 拟合 | 同上 | L2 | 是 |
| FT-006 | Event anomaly 不被普通降噪删除 | `tests/features/test_event_features.py` | L1 | 是 |
| FT-007 | 缺失值 Mask 与填充值分离 | `tests/features/test_missing.py` | L1 | 是 |
| FT-008 | 连续特征均为 finite 或显式失败 | `tests/features/test_numeric_safety.py` | L2 | 是 |
| FT-009 | Stock 60 日窗口不读取未来行 | `tests/features/test_stock_features.py` | L2 | 是 |
| FT-010 | Seat 5/20/60/120 窗口边界正确 | `tests/features/test_seat_features.py` | L2 | 是 |

---

## 8. 标签和可交易状态测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| LB-001 | t+1 Open 为入场价格 | `tests/labels/test_returns.py` | L1 | 是 |
| LB-002 | R1 公式正确 | 同上 | L1 | 是 |
| LB-003 | R3 公式正确 | 同上 | L1 | 是 |
| LB-004 | R5 公式正确 | 同上 | L1 | 是 |
| LB-005 | 五日路径累计与 R1/R3/R5 一致 | 同上 | L2 | 是 |
| LB-006 | 市场超额收益公式正确 | 同上 | L1 | 是 |
| LB-007 | 标签成熟时间正确 | 同上 | L2 | 是 |
| TR-001 | VALID_TRADABLE | `tests/labels/test_tradeability.py` | L1 | 是 |
| TR-002 | SUSPENDED_AT_ENTRY | 同上 | L1 | 是 |
| TR-003 | ONE_PRICE_LIMIT_UP | 同上 | L1 | 是 |
| TR-004 | ONE_PRICE_LIMIT_DOWN | 同上 | L1 | 是 |
| TR-005 | MISSING_ENTRY_PRICE | 同上 | L1 | 是 |
| TR-006 | DELISTED_DURING_HORIZON | 同上 | L1 | 是 |
| TR-007 | Event-study Mask 与 Backtest Mask 分离 | 同上 | L2 | 是 |

---

## 9. Sample、Split、Manifest 和 Seal 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| DS-001 | 活跃 Cycle 每交易日一个 Sample | `tests/dataset/test_samples.py` | L1 | 是 |
| DS-002 | 非交易日不生成 Sample | 同上 | L1 | 是 |
| DS-003 | Sample cutoff 符合披露策略 | 同上 | L2 | 是 |
| DS-004 | 同一 Cycle 不跨 split | `tests/dataset/test_cycle_atomic_split.py` | L2 | 是 |
| DS-005 | Embargo 至少 5 交易日 | 同上 | L2 | 是 |
| DS-006 | Blind 范围与开发范围分离 | 同上 | L2 | 是 |
| DS-007 | Manifest Hash 确定性 | `tests/dataset/test_manifest.py` | L2 | 是 |
| DS-008 | 任一 PIT 违规阻止 Seal | `tests/dataset/test_seal.py` | L2 | 是 |
| DS-009 | 任一 ID 泄漏阻止 Seal | 同上 | L2 | 是 |
| DS-010 | 任一 Cycle 跨切分阻止 Seal | 同上 | L2 | 是 |
| DS-011 | 任一标签一致性错误阻止 Seal | 同上 | L2 | 是 |
| DS-012 | 产物 Hash 不匹配阻止训练 | 同上 | L2 | 是 |

---

## 10. 图 Schema 与张量化测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| GR-001 | 节点类型完整 | `tests/graph/test_schema.py` | L1 | 是 |
| GR-002 | 关系类型完整 | 同上 | L1 | 是 |
| GR-003 | Market 为 graph context 而非超级节点 | 同上 | L1 | 是 |
| GR-004 | 边连续特征排除 ID/Hash/未来信息 | 同上 | L2 | 是 |
| GR-005 | target/context/loss/backtest Mask 独立 | `tests/graph/test_tensorize.py` | L2 | 是 |
| GR-006 | Snapshot round-trip 保持节点和边 | 同上 | L2 | 是 |
| GR-007 | 稳定局部索引确定性 | 同上 | L2 | 是 |
| GR-008 | 悬空边被拒绝 | 同上 | L2 | 是 |
| GR-009 | Top-K 每关系独立 | `tests/graph/test_topk.py` | L2 | 是 |
| GR-010 | 图批处理不混淆 target stock index | 同上 | L2 | 是 |

---

## 11. Base 模型和 OOF 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| MB-001 | Base 输入不接受 Seat/Event/Graph | `tests/models/test_base_shapes.py` | L4 | 是 |
| MB-002 | Base 输出 path[5] 和 regime | 同上 | L4 | 是 |
| MB-003 | Base 累计收益来自 path | 同上 | L4 | 是 |
| MB-004 | 前向输出 finite | 同上 | L4 | 是 |
| OOF-001 | OOF 行不由包含自身 Cycle 的模型生成 | `tests/training/test_oof_independence.py` | L5 | 是 |
| OOF-002 | Fold scaler 不共享未来统计量 | 同上 | L5 | 是 |
| OOF-003 | OOF Manifest 可审计训练边界 | 同上 | L5 | 是 |
| MB-005 | Base tiny batch loss 下降至少 95% | `tests/training/test_base_micro_overfit.py` | L4 | 是 |
| MB-006 | Checkpoint 保存和恢复输出一致 | `tests/training/test_base_checkpoint.py` | L4 | 是 |

---

## 12. Seat、Event 与 Direct 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| SE-001 | 未成熟记录不影响 Seat embedding | `tests/models/test_seat_memory.py` | L4 | 是 |
| SE-002 | 5/20/60/120 窗口 Mask 正确 | 同上 | L4 | 是 |
| SE-003 | 冷启动席位输出 finite | 同上 | L4 | 是 |
| EV-001 | participation 行重排保持事件表示 | `tests/models/test_event_encoder.py` | L4 | 是 |
| EV-002 | 买卖冲突输入影响 conflict score | 同上 | L4 | 是 |
| EV-003 | 空 participation 被显式拒绝 | 同上 | L4 | 是 |
| DR-001 | Direct 输出字段和 shape 正确 | `tests/models/test_direct_model.py` | L4 | 是 |
| DR-002 | Direct 累计收益来自 path | 同上 | L4 | 是 |
| DR-003 | Seat Residual 使用 OOF Base | `tests/training/test_direct_targets.py` | L5 | 是 |
| DR-004 | Amount/Type/Anonymous/Real/Shuffled 配置可复现 | `tests/training/test_direct_baselines.py` | L5 | 是 |
| DR-005 | Direct tiny batch micro-overfit | `tests/training/test_direct_micro_overfit.py` | L4 | 是 |

---

## 13. Relation State 与 Spillover 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| RS-001 | activation 范围 [0,1] | `tests/models/test_relation_state.py` | L4 | 是 |
| RS-002 | strength 非负 | 同上 | L4 | 是 |
| RS-003 | direction 范围 [-1,1] | 同上 | L4 | 是 |
| RS-004 | age 增加导致可学习衰减路径 | 同上 | L4 | 是 |
| RS-005 | activation=0 时边消息严格为 0 | `tests/models/test_gated_message_passing.py` | L4 | 是 |
| RS-006 | gate 改变邻居选择 | 同上 | L4 | 是 |
| RS-007 | gate 改变注意力和消息权重 | 同上 | L4 | 是 |
| SP-001 | Spillover 输出字段和 shape 正确 | `tests/models/test_spillover_model.py` | L4 | 是 |
| SP-002 | Direct target 不进入 Spillover Loss | 同上 | L4 | 是 |
| SP-003 | Spillover candidate 外股票不参与 Loss | 同上 | L4 | 是 |
| SP-004 | Spillover tiny batch micro-overfit | `tests/training/test_spillover_micro_overfit.py` | L4 | 是 |

---

## 14. SeatFlow-FDG 联合模型测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| FDG-001 | final_path 精确等于三分量之和 | `tests/models/test_fdg_output.py` | L4 | 是 |
| FDG-002 | R1/R3/R5 从 final_path 累计 | 同上 | L4 | 是 |
| FDG-003 | 输出 lineage、uncertainty 和 tradeability | 同上 | L4 | 是 |
| FDG-004 | 所有核心模块获得 finite gradient | `tests/models/test_fdg_gradients.py` | L4 | 是 |
| FDG-005 | Direct/Spillover/Event/Backtest Mask 隔离 | `tests/training/test_loss_masks.py` | L4 | 是 |
| FDG-006 | Ranking 仅同日比较 | 同上 | L4 | 是 |
| FDG-007 | NaN/Inf Loss 立即失败 | `tests/training/test_numeric_fail_closed.py` | L4 | 是 |
| FDG-008 | Full tiny batch micro-overfit | `tests/training/test_fdg_micro_overfit.py` | L4 | 是 |
| FDG-009 | 8GB memory plan 可执行前后向和恢复 | `tests/training/test_memory_plan.py` | L4 | 是 |

---

## 15. 评估、Placebo、校准和 Blind 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| MT-001 | MAE/Huber 参考值 | `tests/evaluation/test_metrics.py` | L5 | 是 |
| MT-002 | AUC 参考值 | 同上 | L5 | 是 |
| MT-003 | IC/RankIC 参考值 | 同上 | L5 | 是 |
| MT-004 | NDCG@K/Hit@K 参考值 | 同上 | L5 | 是 |
| MT-005 | Edge AUPR 参考值 | 同上 | L5 | 是 |
| MT-006 | Brier/ECE 参考值 | 同上 | L5 | 是 |
| WF-001 | Walk-Forward 时间顺序 | `tests/evaluation/test_walk_forward.py` | L5 | 是 |
| WF-002 | Cycle 原子和 Embargo | 同上 | L5 | 是 |
| PL-001 | Shuffled Seat 保持样本规模 | `tests/evaluation/test_placebo.py` | L5 | 是 |
| PL-002 | Shuffled Graph 保持度约束 | 同上 | L5 | 是 |
| PL-003 | Placebo 对同 Seed 可复现 | 同上 | L5 | 是 |
| BL-001 | 开发模式拒绝 Blind 标签 | `tests/evaluation/test_blind_access.py` | L5 | 是 |
| BL-002 | Candidate Hash 不匹配拒绝 Blind | 同上 | L5 | 是 |
| BL-003 | Blind 报告不可覆盖 | 同上 | L5 | 是 |
| MG-001 | Direct Gate 失败阻止 Promotion | `tests/evaluation/test_model_gate.py` | L5 | 是 |
| MG-002 | Spillover Gate 失败阻止传播声明 | 同上 | L5 | 是 |
| MG-003 | 经济 Gate 失败阻止模拟盘 | 同上 | L5 | 是 |
| MG-004 | 校准 Gate 失败阻止 Promotion | 同上 | L5 | 是 |

---

## 16. 回测和模拟盘测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| BT-001 | 佣金最低收费 | `tests/backtest/test_costs.py` | L5 | 是 |
| BT-002 | 卖出印花税 | 同上 | L5 | 是 |
| BT-003 | 滑点和冲击成本 | 同上 | L5 | 是 |
| BT-004 | t+1 Open 入场 | `tests/backtest/test_execution.py` | L5 | 是 |
| BT-005 | 停牌不成交 | 同上 | L5 | 是 |
| BT-006 | 一字涨停买单失败 | 同上 | L5 | 是 |
| BT-007 | 一字跌停卖单挂起 | 同上 | L5 | 是 |
| BT-008 | 部分成交和撤单状态 | 同上 | L5 | 是 |
| BT-009 | 连续 Top-K 信号资金约束 | `tests/backtest/test_portfolio.py` | L5 | 是 |
| BT-010 | 持有期重叠正确 | 同上 | L5 | 是 |
| BT-011 | 成本前后结果同时报告 | `tests/backtest/test_report.py` | L5 | 是 |
| PT-001 | 未 Promotion 模型不能开模拟盘 | `tests/backtest/test_paper_account.py` | L5 | 是 |
| PT-002 | 无效模型不能开模拟盘 | 同上 | L5 | 是 |
| PT-003 | 订单记录模型和 Dataset 版本 | 同上 | L5 | 是 |

---

## 17. Python API 与 Worker 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| PA-001 | Health 不泄露 Secret | `tests/api/test_health.py` | L3 | 是 |
| PA-002 | API 符合 Research OpenAPI | `tests/api/test_contract.py` | L3 | 是 |
| PA-003 | 同幂等键返回同 taskId | `tests/tasks/test_idempotency.py` | L3 | 是 |
| PA-004 | 重复请求不重复入队 | 同上 | L3 | 是 |
| PA-005 | Queue payload 不含 DataFrame/Graph/Checkpoint | `tests/tasks/test_queue_payload.py` | L3 | 是 |
| PA-006 | 合法状态转换 | `tests/tasks/test_state.py` | L3 | 是 |
| PA-007 | 终态不可离开 | 同上 | L3 | 是 |
| PA-008 | 取消和 Heartbeat | `tests/tasks/test_worker.py` | L3 | 是 |
| PA-009 | Artifact Hash 校验 | `tests/tasks/test_artifacts.py` | L3 | 是 |
| PA-010 | MinIO 失败返回可重试错误 | 同上 | L3 | 是 |

---

## 18. Java Control Center 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| JC-001 | Task 状态机合法转换 | `TaskServiceTest` | L3 | 是 |
| JC-002 | 终态拒绝后续转换 | 同上 | L3 | 是 |
| JC-003 | 乐观锁防止并发覆盖 | 同上 | L3 | 是 |
| JC-004 | 幂等请求不重复创建任务 | 同上 | L3 | 是 |
| JC-005 | Controller 符合 OpenAPI | `TaskControllerTest` | L3 | 是 |
| JC-006 | Invalid Dataset 阻止 formal experiment | `DatasetGateTest` | L3 | 是 |
| JC-007 | Invalid Model 阻止 Promotion | `ModelPromotionTest` | L3 | 是 |
| JC-008 | 未 Promotion 模型阻止 Paper Session | `PaperSessionTest` | L3 | 是 |
| JC-009 | Dataset Seal 权限 | `RbacTest` | L3 | 是 |
| JC-010 | Blind Access 权限 | 同上 | L3 | 是 |
| JC-011 | Model Promotion 权限 | 同上 | L3 | 是 |
| JC-012 | Paper Activation 权限 | 同上 | L3 | 是 |
| JC-013 | SSE Last-Event-ID 重放 | `TaskSseControllerTest` | L3 | 是 |
| JC-014 | 审计包含 actor/old/new/reason/evidence | `AuditServiceTest` | L3 | 是 |
| JC-015 | Java 未实现标签或回测公式 | `ArchitectureTest` | L1 | 是 |

---

## 19. Next.js 测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| UI-001 | API Error 保留 code/trace/task/retryable | `tests/api-client.test.ts` | L3 | 是 |
| UI-002 | Loading 状态 | 各 feature 测试 | L3 | 是 |
| UI-003 | Empty 状态 | 同上 | L3 | 是 |
| UI-004 | Error 状态 | 同上 | L3 | 是 |
| UI-005 | Degraded 状态 | 同上 | L3 | 是 |
| UI-006 | Synthetic 显示非正式 | `tests/status-badges.test.tsx` | L3 | 是 |
| UI-007 | Invalid 模型不得显示绿色正式状态 | 同上 | L3 | 是 |
| UI-008 | 预测显示 Dataset/Model/cutoff/cost/tradeability | `tests/prediction-page.test.tsx` | L3 | 是 |
| UI-009 | Base/Direct/Spillover/Final 分解 | 同上 | L3 | 是 |
| UI-010 | 传播路径显示 horizon/relation/probability/direction/strength/evidence | `tests/graph-page.test.tsx` | L3 | 是 |
| UI-011 | SSE 重连不重复指标 | `tests/task-stream.test.ts` | L3 | 是 |
| UI-012 | 刷新恢复任务状态 | 同上 | L3 | 是 |
| UI-013 | 无成本配置不得标记正式成本后结果 | `tests/backtest-page.test.tsx` | L3 | 是 |
| UI-014 | 无效模型禁用模拟盘按钮并展示原因 | `tests/paper-page.test.tsx` | L3 | 是 |
| UI-015 | 权限不足隐藏并阻止受控操作 | `tests/rbac-ui.test.tsx` | L3 | 是 |

---

## 20. 可观测性与安全测试

| ID | 测试 | 目标文件 | 类型 | 硬门禁 |
|---|---|---|---|---|
| OB-001 | Trace 包含 taskId/dataset/run/model | `tests/observability/test_trace_context.py` | L3 | 是 |
| OB-002 | API、Worker 和 Java Trace 可关联 | 集成测试 | L3 | 是 |
| OB-003 | 指标包含 queue、task、data、quality、GPU、fold/seed、API、SSE | `infra/tests/test_metrics.py` | L3 | 是 |
| SC-001 | Secret 不出现在 API | API 测试 | L3 | 是 |
| SC-002 | Secret 不出现在日志 | 日志集成测试 | L3 | 是 |
| SC-003 | Blind 标签访问被审计 | Java/Python 集成测试 | L3 | 是 |
| SC-004 | Dataset Seal、Promotion、Paper 激活均审计 | Java 测试 | L3 | 是 |
| SC-005 | 对象存储路径无目录穿越 | Artifact 测试 | L3 | 是 |

---

## 21. E2E 测试

| ID | 场景 | 类型 | 硬门禁 |
|---|---|---|---|
| E2E-001 | 三个应用和基础设施健康 | L6 | 是 |
| E2E-002 | Bronze → Identity → Event → Cycle | L6 | 是 |
| E2E-003 | Feature → Label → Sample → Seal | L6 | 是 |
| E2E-004 | Graph → Tensor → Base OOF | L6 | 是 |
| E2E-005 | Direct → Relation State → Spillover → Full FDG | L6 | 是 |
| E2E-006 | Evaluation → Prediction → Java API → UI | L6 | 是 |
| E2E-007 | Invalid Dataset 在 Java 和 UI 被阻断 | L6 | 是 |
| E2E-008 | Invalid Model 在 Paper Trading 被阻断 | L6 | 是 |
| E2E-009 | SSE 断线重连恢复任务 | L6 | 是 |
| E2E-010 | 全链路 Trace 可查询 | L6 | 是 |

合成 E2E 固定声明：

```text
SYNTHETIC_NON_FORMAL
```

禁止将 E2E 的训练指标或回测结果展示为正式模型结果。

---

## 22. 正式证据门禁

| Gate | 必需证据 | 失败状态 |
|---|---|---|
| Dataset Gate | Seal、PIT=0、ID leakage=0、maturity=0、cycle split=0、label consistency=0 | `dataset_valid=false` |
| Direct Gate | Real Seat 胜过 Amount、Type、Anonymous、Shuffled，且 Base+Direct 胜过 Base | `direct_valid=false` |
| Spillover Gate | Real Graph 胜过 No Graph 和 Shuffled Graph；Edge AUPR 超随机；RankIC 多折多 Seed 稳定 | `spillover_valid=false` |
| Economic Gate | 成本后 Top-K 超额为正，多数 Fold/Seed 为正，回撤和换手可解释 | `economic_valid=false` |
| Calibration Gate | Brier、ECE、Reliability 和 uncertainty coverage 达标 | `calibration_valid=false` |
| Blind Gate | Candidate Hash 完整、Blind 未提前访问、报告不可覆盖 | `blind_valid=false` |
| Promotion Gate | 上述必要 Gate 全部通过并有审批 | `promotion_allowed=false` |
| Paper Gate | Promoted Model 且 `paper_trading_allowed=true` | `paper_trading_allowed=false` |

---

## 23. 每阶段必跑命令

### Contracts

```bash
pytest contracts/tests -v
```

### Python

```bash
ruff check research-engine/src research-engine/tests
ruff format --check research-engine/src research-engine/tests
mypy research-engine/src
pytest research-engine/tests -q
```

### Java

```bash
mvn -f control-center/pom.xml test
mvn -f control-center/pom.xml verify
```

### Frontend

```bash
npm --prefix research-console run lint
npm --prefix research-console run typecheck
npm --prefix research-console test
npm --prefix research-console run build
```

### Infrastructure

```bash
docker compose -f infra/compose/docker-compose.yml config
pytest infra/tests -v
```

### Release Candidate

```bash
python tools/release/verify_release.py
python tools/e2e/run_synthetic_e2e.py
npm --prefix research-console run test:e2e
```

---

## 24. 覆盖率和质量阈值

初始工程阈值：

- Python 行覆盖率 ≥ 85%，核心金融规则分支覆盖率 ≥ 95%。
- Java 行覆盖率 ≥ 80%，审批、状态机和权限分支覆盖率 ≥ 95%。
- Frontend 业务 Adapter、状态组件和门禁组件分支覆盖率 ≥ 90%。
- Contract 测试 100% 通过。
- 数据和模型硬门禁测试不允许标记为 flaky 或跳过。
- Blind、PIT、Cycle、OOF、成本和 Model Gate 测试禁止使用 `xfail`。

覆盖率不是替代行为证据；即使覆盖率达标，任何硬门禁失败仍阻止发布。

---

## 25. 缺陷严重级别

| 级别 | 定义 | 处理 |
|---|---|---|
| S0 | 数据泄漏、Blind 泄漏、错误交易执行、真实密钥泄露 | 立即停止全部开发和实验 |
| S1 | Cycle 跨切分、OOF 污染、标签错误、Model Gate 可绕过 | 阻止合并和发布 |
| S2 | 图路由无效、Mask 混淆、成本计算错误、状态机并发错误 | 阻止对应里程碑 |
| S3 | 页面状态错误、监控缺失、非核心性能问题 | 修复后进入下一阶段 |
| S4 | 文案、布局和非阻断易用性问题 | 纳入正常迭代 |

任何 S0/S1 缺陷修复后，必须重新执行受影响的 Dataset、Experiment、Blind 或 Backtest，旧证据不得继续使用。