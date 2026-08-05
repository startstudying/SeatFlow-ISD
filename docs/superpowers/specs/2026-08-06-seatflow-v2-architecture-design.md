# SeatFlow V2 全量重写架构设计

- 状态：Approved Design
- 日期：2026-08-06
- 目标分支：`main`
- 开发模式：Python、Java、Next.js 三层全部推倒重写
- 项目主线：席位—股票动态异构图的龙虎榜资金传播与短期收益预测
- 数据规范基线：SeatFlow-ISD 数据集构建设计 V1.0
- 模型规范基线：SeatFlow × S³G 深度迁移方案 V1.0

---

## 1. 设计结论

SeatFlow V2 不再被定义为“龙虎榜图上的收益预测器”，而被定义为一个完整的事件驱动研究平台：

```text
多源原始数据
→ 点时一致的数据湖
→ Event / Cycle / Sample / Universe
→ Base OOF 与 Seat Residual
→ 事件中心动态异构图
→ 潜在资金关系状态
→ t+1 / t+3 / t+5 未来影响图
→ 状态控制 HGT
→ Direct / Spillover 多任务预测
→ Walk-Forward、密封盲测、成本后回测
→ 模型注册、每日推理、模拟盘与传播解释
```

系统的第一预测对象不是收益，而是未来资金影响关系；收益是资金影响持续、迁移、扩散、衰减或反转后的结果。

本设计同时冻结以下原则：

1. 数据口径优先于模型复杂度。
2. `Cycle` 是训练、验证、测试切分的原子单位。
3. 所有输入必须满足 Point-in-Time 可用性。
4. 未成熟标签不得进入席位历史画像。
5. 关系状态和未来影响图必须直接控制消息路由，不允许只在预测头前拼接。
6. 训练图事实来源为 Iceberg/Parquet；Neo4j 仅作查询投影。
7. Python 是金融研究语义的唯一实现端。
8. Java 负责任务、权限、审批、版本和审计，不复制 Cycle、标签、图、损失或回测逻辑。
9. Next.js 只通过版本化 API 使用平台能力，不直接读取 Python 私有文件或数据库私表。
10. 任意核心证据不完整时必须 Fail-Closed。
11. 单机 8GB GPU 必须能串行完成完整科研流程；服务器只提高速度、并发和可选模型规模。

---

## 2. 范围

### 2.1 本次重写必须交付

- 标准 Monorepo 目录与统一工程规范。
- 多数据源 Bronze 采集与原始血缘。
- Silver 点时实体归一与席位身份版本。
- Event、Cycle、Sample、Universe 全量生命周期。
- Stock、Seat、Event、Theme、Market 特征体系。
- t+1、t+3、t+5 累计市场超额收益及方向标签。
- Base Market 模型、滚动 OOF 预测及 Seat Residual。
- 事件中心动态异构图。
- 关系级资金状态模块。
- Horizon-specific 未来影响图。
- 状态控制的异构图传播。
- Direct 与 Spillover 双任务。
- 收益、方向、排序、未来边、状态约束联合训练。
- Walk-Forward、Embargo、Blind Test、基线与消融。
- 真实交易约束、成本归因、模拟盘。
- Python Research API 与 Python Temporal Worker。
- Spring Boot Control API 与 Java Temporal Worker。
- Next.js Research Console。
- Docker Compose 与 k3s 双环境。
- PostgreSQL、Redis、MinIO、Iceberg、MLflow、Neo4j、Temporal、OpenTelemetry。
- OpenAPI、gRPC、SSE、领域 JSON Schema 和生成客户端。
- 单元、契约、集成、数据、模型、工作流、端到端测试。
- Manifest、Lineage、Quality Gate、Model Gate 和 Deployment Gate。

### 2.2 明确不进入 V2 首版核心范围

- 智能体交易决策。
- 自动真实券商下单。
- 无限多跳传播。
- 未经验证的因果结论。
- 高频盘口和毫秒级执行。
- 强化学习交易策略。
- 新闻、社媒、公告文本大模型融合。
- 大规模全连接未来邻接矩阵。
- 把 Neo4j 当作训练主存储。

这些扩展不得阻塞 V2 核心研究闭环。

---

## 3. 旧工程处理与安全边界

用户已确认直接在 `main` 上重写，不创建长期开发分支。正式实施前必须完成：

1. 创建不可变备份 Tag：`legacy-r4-before-isd-esd-rewrite`。
2. 记录 Tag 对应 Git SHA、当前测试结果和关键 Artifact Hash。
3. 将仍有参考价值的旧实现迁入 `archive/legacy-r4/` 或通过 Tag 保留。
4. 旧模块不得继续作为新模块的隐式依赖。
5. 新系统不得继承旧系统的“已完成”“模型有效”“盲测有效”状态。
6. 旧数据、旧 Checkpoint 和旧评估产物只能以 `LEGACY_UNVERIFIED` 身份导入。

旧代码可作为迁移参考的部分包括：

- 数据提供方适配经验。
- PIT、防泄漏和实体 ID 测试。
- PyG 异构图张量化经验。
- Edge-Gated Relation、训练器、回测和模拟盘实现经验。
- Java API、审计、设置和数据中心模块经验。
- Next.js 页面和图可视化经验。

禁止复制旧目录结构作为新架构基础。

---

## 4. Monorepo 目标结构

```text
SeatFlow/
├── apps/
│   ├── web/                         # Next.js Research Console
│   └── control-api/                 # Spring Boot Control Plane
│
├── services/
│   └── research-engine/             # Python Research Engine
│       ├── pyproject.toml
│       ├── src/seatflow/
│       │   ├── ingestion/
│       │   ├── lakehouse/
│       │   ├── identity/
│       │   ├── lifecycle/
│       │   ├── universe/
│       │   ├── features/
│       │   ├── labels/
│       │   ├── graph/
│       │   ├── models/
│       │   ├── training/
│       │   ├── evaluation/
│       │   ├── backtest/
│       │   ├── inference/
│       │   ├── workflow/
│       │   ├── api/
│       │   └── observability/
│       └── tests/
│
├── packages/
│   ├── api-contracts/               # OpenAPI / AsyncAPI / Proto
│   ├── domain-schemas/              # JSON Schema / Avro / Iceberg Schema
│   ├── generated-clients/           # Java / Python / TypeScript clients
│   └── ui-kit/                      # Design tokens and reusable UI
│
├── datasets/
│   └── seatflow-isd/
│       ├── schemas/
│       ├── manifests/
│       ├── quality-reports/
│       └── local-catalog/
│
├── experiments/
│   ├── configs/
│   ├── protocols/
│   ├── reports/
│   └── ablations/
│
├── infra/
│   ├── compose/
│   ├── helm/
│   ├── k3s/
│   ├── database/
│   ├── observability/
│   └── scripts/
│
├── tools/
│   ├── codegen/
│   ├── migrations/
│   ├── data-audit/
│   └── release/
│
├── docs/
│   ├── architecture/
│   ├── adr/
│   ├── runbooks/
│   ├── research/
│   └── superpowers/specs/
│
└── archive/
    └── legacy-r4/
```

### 4.1 依赖方向

```text
packages/domain-schemas
        ↓
packages/api-contracts
        ↓
services/research-engine  ←→  apps/control-api
        ↓                         ↓
        └──────────────→ apps/web
```

强制规则：

- `apps/web` 不得依赖 Python 源代码。
- `apps/control-api` 不得依赖 Python 内部模块。
- Python 和 Java 通过契约通信，不通过共享数据库私表通信。
- `packages/domain-schemas` 是跨语言领域对象的唯一语义来源。
- 生成客户端不得手工修改。

---

## 5. 三层职责

## 5.1 Python Research Engine

负责所有研究与金融语义：

- 数据采集与原始落湖。
- 数据源修订与冲突处理。
- 股票、席位、行业、概念实体归一。
- PIT 查询。
- Event、Cycle、Sample、Universe。
- 交易日历与交易可执行性。
- 特征注册、构建、归一化和缺失掩码。
- 标签、日级路径和成熟标签。
- Base OOF 与 Seat Residual。
- 图候选、边构建、图快照和张量化。
- 模型、损失、训练、评估、消融、解释。
- 回测、成本模型和模拟盘研究逻辑。
- Python Temporal Activities。
- Research REST API 与内部 gRPC 服务。

Python 不负责：

- 用户管理。
- RBAC。
- 审批流程。
- 系统设置 UI。
- 跨团队业务审计展示。

## 5.2 Spring Boot Control Plane

负责平台治理：

- 用户、角色、权限、API Key。
- 数据源配置与密钥引用。
- 数据集构建请求。
- 实验配置登记与运行发起。
- Temporal Workflow 启动、查询、Signal、Cancel。
- 数据集版本、模型版本、部署版本元数据。
- 质量门禁、模型门禁、发布门禁状态。
- 人工审批。
- 审计日志。
- 通知。
- SSE 事件聚合。
- 对前端提供统一 API。
- Java Temporal Activities。

Java 禁止实现：

- Cycle 延长逻辑。
- 标签公式。
- Seat Residual。
- 图候选生成。
- 模型损失。
- 交易收益计算。
- 研究指标计算。

## 5.3 Next.js Research Console

至少包含：

1. 总览驾驶舱。
2. 数据源管理。
3. Dataset Build 控制台。
4. 数据质量中心。
5. Event/Cycle/Sample 浏览器。
6. 席位身份与别名审核。
7. 特征注册表。
8. 图快照浏览。
9. 资金传播图探索。
10. 实验创建与训练监控。
11. 实验比较。
12. 未来影响图评估。
13. Direct/Spillover 预测页。
14. 单股票解释页。
15. 回测与成本归因。
16. 密封盲测审批页。
17. 模型注册与发布。
18. 每日推理状态。
19. 模拟盘。
20. 系统设置、用户、权限、审计与运行监控。

前端必须明确显示：

- 数据是否完整。
- 数据集是否有效。
- 模型是否有效。
- 是否是 Blind 结果。
- 是否扣除交易成本。
- 是否可交易。
- 是否为降级数据源。

禁止使用合成结果伪装正式结果。

---

## 6. 数据基础设施

## 6.1 对象存储与 Lakehouse

- MinIO：本地和服务器统一 S3-compatible 接口。
- Parquet：列式物理文件。
- Apache Iceberg：表、Schema 演进、快照、分区与时间旅行。

逻辑层：

```text
bronze → silver → gold → graph → sample
```

### Bronze

保留源数据原貌，必备字段：

- `source`
- `source_record_id`
- `event_time`
- `available_time`
- `ingested_at`
- `source_version`
- `raw_payload_hash`
- `revision_no`
- `ingestion_run_id`
- `source_confidence`

### Silver

- `dim_stock_pit`
- `dim_seat_identity_pit`
- `dim_seat_alias_pit`
- `dim_theme_pit`
- `fact_lhb_event`
- `fact_lhb_participation`
- `fact_stock_theme_pit`
- `fact_tradeability`
- `fact_cycle`

### Gold

- `feat_stock_daily`
- `feat_seat_daily`
- `feat_lhb_event`
- `feat_theme_daily`
- `market_context_daily`
- `edge_seat_event`
- `edge_event_stock`
- `edge_event_event`
- `edge_stock_stock`
- `edge_stock_theme`
- `sample_universe`
- `label_stock_return`
- `base_oof_prediction`
- `seat_residual_target`

### Graph

- 每日增量节点状态。
- 当日新增、更新、失效边。
- 节点稳定 ID 映射。
- Target/Context Mask。
- Market Context。
- 图 Schema 版本。

### Sample

- `stock_sequence_60d`
- `graph_sequence_20d`
- `market_context`
- `target_stock_index`
- `target_mask`
- `context_mask`
- `loss_mask`
- `backtest_mask`
- `labels`
- `metadata`

## 6.2 PostgreSQL

仅保存控制面和小型元数据：

- users
- roles
- permissions
- api_keys
- data_sources
- dataset_versions
- dataset_builds
- quality_gate_results
- experiments
- experiment_runs
- model_versions
- model_promotions
- workflow_runs
- approvals
- deployment_records
- notifications
- audit_logs

不得把大规模训练样本和图张量长期存 PostgreSQL。

## 6.3 Redis

- 分布式锁。
- 短期 API 缓存。
- SSE 消费游标。
- 非权威任务进度缓存。
- 限流。

Temporal 和 PostgreSQL 才是工作流权威状态，Redis 不得成为唯一状态来源。

## 6.4 MLflow

- 参数。
- 指标。
- Checkpoint 引用。
- 图表和报告。
- Dataset Manifest Hash。
- Git SHA。
- Environment Lock Hash。
- Model Registry。

正式训练时 MLflow 不可用应 Fail-Closed。Smoke 模式可使用本地简化追踪器，但必须标记为 `NON_FORMAL_RUN`。

## 6.5 Neo4j

仅用于：

- 前端关系查询。
- 传播路径检索。
- 关键子图展示。
- 席位、事件、股票、主题的交互式探索。

Neo4j 数据由已封存图产物投影生成，可重建，不是训练事实来源。

---

## 7. 多数据源接入

支持：

- Tushare。
- 交易所公开披露。
- 东方财富公开适配器。
- 同花顺公开适配器。
- 授权历史数据文件。
- AKShare 交叉校验和降级源。

### 7.1 多源处理原则

1. 所有来源先独立进入 Bronze。
2. 原始记录不可跨来源覆盖。
3. 同一事件以业务键和源版本归并。
4. 冲突必须生成冲突报告。
5. 选择结果必须保留规则版本和证据。
6. 席位明细缺失时不得由汇总金额推导虚假席位。
7. 数据源权限不足必须明确标记 `DEGRADED`。
8. 历史修订通过 `revision_no` 和新 Dataset Version 处理。

### 7.2 数据源适配器接口

每个适配器实现：

- `probe_capabilities()`
- `fetch_stock_master()`
- `fetch_trade_calendar()`
- `fetch_daily_bar()`
- `fetch_daily_basic()`
- `fetch_lhb_summary()`
- `fetch_lhb_participation()`
- `fetch_industry_membership()`
- `fetch_concept_membership()`
- `fetch_tradeability()`
- `get_availability_policy()`

返回值必须含来源时间和可用时间，不接受只有业务值而无时间血缘的正式记录。

---

## 8. 核心领域对象

## 8.1 Event

一次股票在一个交易日、一个披露批次上的完整龙虎榜事件：

```text
event_id = event_date + stable_stock_id + disclosure_batch
```

同日多个买方、卖方席位保留独立参与事实。Event 不得压缩为单条 Seat→Stock 净额边。

## 8.2 Cycle

首次上榜创建周期：

```text
cycle_start = first_lhb_date
cycle_end = latest_lhb_date + 5 个交易日
```

若新事件日期不晚于当前 `cycle_end`：

- 归入同一 `cycle_id`。
- 新建 Event。
- `cycle_end` 重置为新事件日后第 5 个交易日。

若新事件在当前 `cycle_end` 之后，创建新 Cycle。

## 8.3 Sample

Cycle 内每个交易日生成一个滚动预测样本：

- `sample_id`
- `cycle_id`
- `stock_id`
- `prediction_date`
- `cutoff_time`
- `days_since_last_lhb`
- `target_pool_reason`
- `dataset_version`
- `feature_schema_hash`
- `label_policy_hash`

## 8.4 Universe

- Target Stock：处于活跃 Cycle，参与损失、排名和回测。
- Context Stock：通过席位、行业、概念或传播关系进入局部图，只参与消息传递。
- Base Universe：全市场合格股票，用于学习普通市场收益。

主预测池：

```text
0 <= days_since_last_lhb <= 5
```

Context 候选必须按关系强度和数量截断，默认最多两跳传播。

---

## 9. Point-in-Time 与防泄漏

所有正式表和特征必须区分：

- `event_time`
- `available_time`
- `effective_from`
- `effective_to`
- `ingested_at`

### 9.1 输入规则

对任意 Sample：

```text
input.available_time <= sample.cutoff_time
```

### 9.2 成熟标签

席位历史收益、胜率和能力只能纳入：

```text
label_end_time <= current_cutoff_time
```

### 9.3 禁止项

- 未来是否再次上榜。
- 未来席位进入或退出。
- 未来停牌或涨跌停。
- 用最新席位身份修订覆盖全部历史。
- 使用全样本归一化参数。
- 同一 Cycle 跨切分。
- 用 Base 模型训练内拟合值生成 Seat Residual。
- 数值身份 Hash 进入连续特征。

### 9.4 特征注册表

每个特征必须声明：

- entity_type
- source_tables
- lookback_window
- availability_rule
- maturity_rule
- transform
- normalization
- missing_policy
- dtype
- continuous/discrete
- train_allowed
- lineage_version

---

## 10. 标签

决策时点：t 日收盘且龙虎榜披露完成后。

入场：t+1 开盘。

退出：t+1、t+3、t+5 收盘。

```text
R1 = Close[t+1] / Open[t+1] - 1
R3 = Close[t+3] / Open[t+1] - 1
R5 = Close[t+5] / Open[t+1] - 1
```

主标签：

```text
market_excess_return_h = stock_return_h - market_return_h
h ∈ {1, 3, 5}
```

辅助标签：

- raw return。
- industry excess return。
- direction。
- daily return path。
- tail risk。
- future edge labels。

日级路径累计结果必须与 1/3/5 日累计标签严格一致。

### 10.1 可交易状态

- VALID_TRADABLE
- SUSPENDED_AT_ENTRY
- ONE_PRICE_LIMIT_UP
- ONE_PRICE_LIMIT_DOWN
- MISSING_ENTRY_PRICE
- DELISTED_DURING_HORIZON
- CORPORATE_ACTION_SPECIAL

事件研究样本与策略回测样本使用不同 Mask，不得删除所有不可交易事件后再报告事件效果。

---

## 11. Base OOF 与 Seat Residual

### 11.1 Base Model

作用：学习股票在没有席位图信息时“本来会怎样”。

输入：

- 量价。
- 市场环境。
- 行业/概念。
- 流动性、风险、非理性状态。

### 11.2 OOF

Base 预测必须通过滚动折外方式生成：

```text
base_oof_prediction_h
```

### 11.3 Residual

```text
seat_residual_target_h
= true_market_excess_h - base_oof_prediction_h
```

完整 SeatFlow 模型可同时预测：

- 最终超额收益。
- 席位增量收益。
- 方向。
- 排名。

---

## 12. 图定义

### 12.1 节点

- Seat
- LHBEvent
- Stock
- Theme（Industry/Concept 统一实体，保留类型）

Market Context 作为全局上下文，不作为全连接超级节点。

### 12.2 关系

- seat_buy_event
- seat_sell_event
- event_impacts_stock
- stock_belongs_industry
- industry_contains_stock
- stock_has_concept
- concept_contains_stock
- same_stock_repeat
- same_seat_relay
- same_coalition_relay
- same_theme_diffusion
- capital_rotation
- seat_relay
- coalition_relay
- theme_diffusion
- return_comovement_aux

### 12.3 边特征原则

边可包含：

- amount
- amount_share
- rank
- amount_surprise
- identity_confidence
- lag_days
- direction
- repeated_seat_ratio
- coalition_similarity
- relation_confidence
- event_age
- decay_state

边不得包含：

- stock_id 数值。
- seat_id 数值。
- event_id 数值。
- ID Hash。
- 绝对日期整数。
- 未来收益。
- 未来再次上榜状态。

---

## 13. 模型架构

正式模型名：`SeatFlow-ESD`。

### 13.1 编码层

- Stock Sequence Encoder：60 日量价与风险序列。
- Seat Profile Encoder：5/20/60/120 日历史画像。
- Event Encoder：资金规模、结构、价格环境、延续与博弈摘要。
- Theme Encoder：主题热度、扩散、拥挤和生命周期。
- Market Context Encoder。

### 13.2 双通道特征处理

基础市场通道：

- 可进行趋势分解和适度降噪。

事件异常通道：

- 保留净买入突变、换手突变、集中度、涨跌停和机构占比。
- 仅做 `log1p`、Robust Scaling 和训练折内裁剪。

事件异常不得被普通降噪流程删除。

### 13.3 关系状态模块

为每种关系类型维护稀疏隐藏状态：

```text
H_t^r = decay(H_{t-1}^r, Δt, market_state)
        + gate_t^r * event_injection_t^r
```

状态语义至少包括：

- 资金冲击强度。
- 持续性。
- 方向反转。
- 席位协同。
- 跨股票迁移。
- 主题扩散。
- 拥挤与退潮。

### 13.4 Future Impact Graph

三个独立 Decoder：

- t+1：直接冲击与隔日兑现。
- t+3：持续、接力与扩散。
- t+5：衰减、拥挤与反转。

输出候选边的：

- 激活概率。
- 影响方向。
- 影响强度。
- 置信度。

禁止生成全市场 N×N 稠密邻接矩阵。先生成经济合理候选边，再 Top-K 解码。

### 13.5 状态控制 HGT

未来影响图必须进入：

- 邻居选择。
- 边激活。
- 消息权重。
- 注意力权重。
- 正负方向门控。

只在最终预测头拼接状态向量不算完成。

### 13.6 双任务

Direct Head：

- 预测当前活跃龙虎榜 Cycle 的目标股票。
- t+1 为主要任务，兼顾 t+3/t+5。

Spillover Head：

- 预测未直接上榜但位于传播路径中的 Context 股票。
- t+3/t+5 为主要任务。

只有 Spillover 稳定有效，项目才可声称建模资金传播。

### 13.7 损失

```text
L = Σ_h w_h (
      L_return_h
    + λ_rank L_rank_h
    + λ_direction L_direction_h
    + λ_graph L_graph_h
    + λ_state L_state_h
    + λ_risk L_risk_h
)
```

- Return：Huber 或 Quantile。
- Rank：Pairwise/Listwise，限制同一预测日期内比较。
- Direction：二分类或三分类。
- Graph：边存在、方向、强度。
- State：衰减、一致性、反转、拥挤约束。
- Risk：尾部风险或不可交易风险。

---

## 14. 训练阶段

推荐训练阶段：

1. 数据契约与质量门禁。
2. 非图 Base 基线。
3. Event Graph 编码预训练。
4. Future Impact Graph 预训练。
5. Relation State 训练。
6. 状态控制 HGT 联合训练。
7. Direct/Spillover 联合微调。
8. Walk-Forward 评估。
9. 消融矩阵。
10. 多随机种子。
11. Blind Candidate Freeze。
12. 人工审批。
13. Blind Evaluation。
14. 模型注册。

不得在数据集未 Seal 前启动正式训练。

---

## 15. 切分与评估

### 15.1 Walk-Forward

每折：

- Train：过去全部成熟周期。
- Validation：后续约 60 个交易日。
- Embargo：至少 5 个交易日。
- Test：再后续约 20 个交易日。

同一 `cycle_id` 必须完整进入同一个集合。

### 15.2 Blind Test

- 最后连续约 6 个月。
- 开发期间不得查询标签指标。
- Blind Manifest、代码 SHA、配置、模型候选在执行前冻结。
- 执行结果不可覆盖。
- 不得根据 Blind 结果继续调参并仍称为 Blind。

### 15.3 指标

收益：

- MAE
- Huber
- Direction Accuracy
- AUC

排序：

- IC
- RankIC
- ICIR
- RankICIR
- NDCG@K
- Hit@K

未来图：

- Edge AUC
- Edge AUPR
- Recall@K
- Direction Accuracy
- Calibration

回测：

- 累计收益
- 年化收益
- Sharpe
- 最大回撤
- 胜率
- 换手率
- 成本前后收益
- 滑点与冲击成本归因

机制：

- Direct 与 Spillover 分组。
- 上榜原因分组。
- 席位类型分组。
- 市场状态分组。
- 主题启动/扩散/退潮分组。

---

## 16. Temporal 工作流

### 16.1 核心工作流

- BuildDatasetWorkflow
- RebuildAffectedPartitionsWorkflow
- TrainBaseModelWorkflow
- GenerateBaseOOFWorkflow
- BuildSeatResidualWorkflow
- PretrainEventGraphWorkflow
- PretrainFutureImpactGraphWorkflow
- TrainSeatFlowESDWorkflow
- EvaluateWalkForwardWorkflow
- RunAblationMatrixWorkflow
- RunMultiSeedWorkflow
- SealBlindCandidateWorkflow
- RunBlindEvaluationWorkflow
- PublishModelWorkflow
- DailyInferenceWorkflow
- PaperTradingWorkflow

### 16.2 Activity 约束

每个 Activity 必须：

- 幂等。
- 可重试。
- 支持 Heartbeat。
- 支持取消。
- 使用轻量输入输出。
- 写入 Manifest。
- 记录 Trace ID。
- 明确 Retryable 与 Non-Retryable Error。

Temporal Payload 只传：

- ID。
- Version。
- URI。
- Hash。
- 小型参数。

不得传 DataFrame、PyG 图、Checkpoint 或大文件。

### 16.3 状态机

```text
REQUESTED
→ RUNNING
→ BLOCKED / FAILED / CANCELLED / COMPLETED
```

质量门禁失败：

```text
dataset_valid = false
training_allowed = false
workflow_status = BLOCKED
```

---

## 17. 通信协议

### 17.1 Next.js → Java

- REST + OpenAPI。
- SSE 实时进度。

### 17.2 Java → Python

- REST + OpenAPI：查询与控制面请求。
- gRPC：内部批量评分和高吞吐推理。
- Temporal SDK：长工作流。

### 17.3 大型产物

统一传递：

```json
{
  "artifactUri": "s3://...",
  "manifestHash": "sha256:...",
  "schemaVersion": "seatflow.domain.v1"
}
```

### 17.4 SSE 事件

- WORKFLOW_STARTED
- ACTIVITY_STARTED
- ACTIVITY_PROGRESS
- QUALITY_GATE_PASSED
- QUALITY_GATE_FAILED
- TRAINING_METRIC
- ARTIFACT_CREATED
- WORKFLOW_BLOCKED
- WORKFLOW_COMPLETED
- WORKFLOW_FAILED

SSE 支持 `Last-Event-ID` 恢复。

### 17.5 统一错误模型

```json
{
  "code": "DATASET_QUALITY_GATE_FAILED",
  "message": "Cycle cross-split violations detected",
  "traceId": "...",
  "workflowId": "...",
  "details": {},
  "retryable": false
}
```

错误码至少覆盖：

- DATA_NOT_AVAILABLE
- SOURCE_PERMISSION_DENIED
- PIT_TIMESTAMP_MISSING
- IDENTITY_RESOLUTION_FAILED
- CYCLE_POLICY_VIOLATION
- LABEL_NOT_MATURE
- DATASET_QUALITY_GATE_FAILED
- TRAINING_NOT_ALLOWED
- CHECKPOINT_NOT_FOUND
- MODEL_NOT_VALID
- BLIND_ACCESS_DENIED
- TRADE_NOT_EXECUTABLE
- WORKFLOW_CANCELLED

---

## 18. 本地 8GB GPU 完整运行模式

### 18.1 Profiles

- `local-smoke`
- `local-full-slow`
- `server-full`

`local-full-slow` 必须支持：

- 全历史 Iceberg 重建。
- 完整 Walk-Forward。
- 全部基线和消融。
- 多随机种子。
- 六个月密封盲测。
- 全市场传播图推理。

方式是串行、分块、断点恢复，不是删功能。

### 18.2 本地资源策略

- GPU Activity 并发：1。
- 正式实验并发：1。
- Fold 串行。
- Seed 串行。
- 关系候选 Top-K。
- 图 Out-of-Core。
- CPU 保存图，按批搬运 GPU。
- AMP。
- Gradient Accumulation。
- Gradient Checkpointing。
- 模型和优化器阶段间释放。
- Windows DataLoader Worker 默认 0–2。

### 18.3 禁止行为

- OOM 后静默切 CPU。
- 自动缩短正式数据区间。
- 自动减少正式 Fold 数量。
- 自动删除正式消融项。
- 自动改变 Blind 区间。
- 不记录的隐式降级。

任何自动显存降级都必须记录实际配置和原因。

### 18.4 断点恢复

所有长任务记录：

- build_id
- activity_id
- partition_id
- experiment_id
- fold_id
- seed
- input_manifest_hash
- output_manifest_hash
- status
- started_at
- completed_at

电脑重启后允许从最近完成点继续。

---

## 19. Docker Compose 与 k3s

### 19.1 本地 Compose Profiles

`core`：

- PostgreSQL
- Redis
- MinIO
- Temporal Server
- Temporal UI
- OpenTelemetry Collector

`app`：

- Web
- Control API
- Java Worker
- Research API
- Python Worker

`research`：

- MLflow

`graph`：

- Neo4j

`observability`：

- Prometheus
- Grafana
- Loki

日常使用 `core + app`；其他按需启动。

### 19.2 k3s Namespaces

- seatflow-app
- seatflow-workflow
- seatflow-data
- seatflow-training
- seatflow-observability

### 19.3 GPU 调度

GPU Worker 使用节点标签和资源限制。ETL、质量检查和普通图构建默认进入 CPU Worker。

### 19.4 配置一致性

Docker Compose 和 k3s：

- 使用相同镜像。
- 使用相同配置键。
- 使用相同 Schema。
- 使用相同迁移。
- 使用相同 Health Contract。

---

## 20. 可观测性

统一 OpenTelemetry：

- HTTP Trace。
- gRPC Trace。
- Temporal Workflow/Activity Trace。
- Dataset Build Trace。
- Training Run Trace。
- Inference Trace。

必须关联：

- trace_id
- workflow_id
- dataset_version
- experiment_run_id
- model_version
- prediction_date

指标至少包含：

- Activity 时长和失败率。
- 数据源延迟。
- 分区构建进度。
- 数据质量指标。
- GPU 显存和利用率。
- Fold/Seed 完成进度。
- API 延迟。
- 推理吞吐。
- SSE 连接状态。

---

## 21. 安全与权限

- 数据源密钥只通过 Secret 注入。
- 前端永远不获取真实数据源 Token。
- Java 保存密钥引用，不在普通 API 返回明文。
- 所有数据集 Seal、Blind 解封、模型 Promote 和部署操作需权限控制。
- Blind 标签访问必须单独审计。
- 所有写操作记录操作者、时间、请求 ID、旧值、新值和结果。
- 对象存储使用最小权限访问策略。

---

## 22. 测试策略

### 22.1 Python

- 单元测试。
- Property-based 测试。
- PIT 泄漏测试。
- Cycle 边界测试。
- 标签公式测试。
- 日级路径一致性测试。
- OOF 独立性测试。
- ID 禁入连续特征测试。
- 图 Schema 测试。
- 模型 Shape/Gradient 测试。
- Micro-overfit 测试。
- 8GB Memory Plan 测试。
- Walk-Forward 原子性测试。
- Blind 访问测试。
- 回测执行规则测试。

### 22.2 Java

- Controller 测试。
- Service 测试。
- Repository 测试。
- RBAC 测试。
- Temporal Workflow 测试。
- 幂等与重复请求测试。
- SSE 恢复测试。
- 审批门禁测试。
- 审计日志测试。

### 22.3 Next.js

- 组件测试。
- API Adapter 测试。
- 空态、错误态、降级态测试。
- SSE 重连测试。
- 权限显示测试。
- 数据状态真实性测试。
- Playwright 端到端测试。

### 22.4 契约测试

- OpenAPI 兼容性。
- Proto 兼容性。
- JSON Schema 兼容性。
- Iceberg Schema 兼容性。
- 生成客户端一致性。

### 22.5 端到端测试

合成数据必须跑通：

```text
ingest
→ silver
→ cycles
→ features
→ labels
→ graph
→ samples
→ quality seal
→ base OOF
→ model smoke
→ evaluation
→ prediction
→ Java API
→ Next.js display
```

---

## 23. 质量门禁

初始门槛：

- 日行情完整率 >= 99.5%。
- 龙虎榜摘要—席位明细关联率 >= 95%。
- 席位归一成功率 >= 90%。
- 中高置信席位比例 >= 85%。
- 行业 PIT 覆盖率 >= 99%。
- 概念 PIT 覆盖率 >= 95%。
- 标签完整率 >= 98%。
- available_time 违规 = 0。
- ID 数值进入连续特征 = 0。
- 未成熟标签进入席位画像 = 0。
- Cycle 跨切分 = 0。

正式训练入口必须读取 Dataset Seal。没有 Seal 或 Seal Invalid 直接退出。

---

## 24. Manifest 与血缘

Dataset Manifest：

- dataset_version
- build_timestamp
- date_range
- source_manifest_hash
- feature_schema_hash
- label_policy_hash
- cycle_policy_hash
- split_definition_hash
- graph_schema_hash
- row_counts
- node_counts
- edge_counts
- quality_gate_results

Experiment Manifest：

- dataset_manifest_hash
- git_sha
- environment_lock_hash
- config_hash
- model_architecture_hash
- seed
- fold_definition
- actual_memory_plan
- checkpoint_uris
- metrics_uris

Prediction Manifest：

- prediction_date
- dataset_version
- model_version
- cutoff_time
- graph_snapshot_hash
- output_uri
- explanation_uri

---

## 25. 前端信息架构原则

每个研究页面必须回答：

1. 数据来自哪里。
2. 截止时间是什么。
3. 使用哪个 Dataset Version。
4. 使用哪个 Model Version。
5. 当前状态是否有效。
6. 是否为正式运行。
7. 是否为 Blind。
8. 是否可交易。
9. 结果如何追溯。

图可视化必须支持从预测股票反查：

```text
Seat → Event → Target Stock → Theme/Relay → Spillover Stock
```

并展示：

- 关系状态。
- 未来边概率。
- 影响方向。
- 影响强度。
- Horizon。
- 证据时间。

---

## 26. 验收标准

### 26.1 数据

- 任意 Sample 可追溯到原始源记录。
- 任意 Event 可找到全部 Seat Participation。
- 任意 Cycle 可恢复全部 Event 和每日 Sample。
- Cycle 延长规则有自动化测试。
- 所有输入满足 available_time。
- 标签和日级路径一致。
- 同一 Cycle 不跨 Split。
- Base OOF 与 Residual 血缘独立。

### 26.2 模型

- 8GB 模式可以执行前向、反向、Checkpoint 和恢复。
- Relation State 直接进入消息路由。
- 三个 Horizon 有独立未来图 Decoder。
- Direct 与 Spillover 均有独立指标。
- Future Edge 有独立监督与评估。
- 完整模型可与简单拼接模型消融比较。

### 26.3 平台

- Java 可启动、查询、取消和恢复 Temporal Workflow。
- 前端可持续显示进度并刷新恢复。
- Quality Gate 失败后训练按钮和 API 均被阻断。
- 模型无有效证据时不展示伪预测。
- 所有操作可通过 Trace ID 串联。

### 26.4 部署

- 本地 Compose 可启动 core/app。
- 完整 Profile 可按需启动。
- k3s Helm 部署通过。
- 同一镜像可运行于 Compose 和 k3s。
- Secret 不进入 Git。

---

## 27. 主要风险与缓解

### 数据源不完整

症状：席位明细不足、未来边标签稀疏。

缓解：多源采集、置信度、降级标记、正样本—未标注学习、严格报告覆盖率。

### 图过密

症状：8GB OOM、消息混杂。

缓解：候选边、关系 Top-K、两跳限制、Out-of-Core、分块。

### 极端值主导

症状：Loss 爆炸、Logit 极端。

缓解：log1p、RobustScaler、折内裁剪、梯度裁剪、事件通道保护。

### 只学到市场 Beta

症状：牛市有效，其他状态失效。

缓解：Base OOF、市场超额收益、市场状态分层、Placebo/Shuffled 图。

### 未来图不可解释

症状：收益有提升但边预测差。

缓解：独立边监督、状态约束、路径一致性、图指标门禁。

### 本地运行时间过长

症状：完整实验持续数日。

缓解：断点恢复、Fold/Seed 队列、进度估算、任务优先级、服务器无缝迁移。

### Direct 有效而 Spillover 无效

症状：只能证明龙虎榜事件效应。

缓解：单独 Spillover Universe、独立损失、未上榜股票指标和成本后结果。

---

## 28. 设计决策记录

- ADR-001：直接重写 `main`，但先创建 Legacy Tag。
- ADR-002：Python、Java、Next.js 三层全量重写。
- ADR-003：研究平台型职责边界。
- ADR-004：标准 Monorepo。
- ADR-005：Iceberg + Parquet + MinIO Lakehouse。
- ADR-006：PostgreSQL 控制面、Redis 缓存、MLflow 实验、Neo4j 投影。
- ADR-007：Temporal 持久化工作流。
- ADR-008：本地 Compose + 服务器 k3s。
- ADR-009：REST + SSE + gRPC + Temporal + Artifact URI。
- ADR-010：多数据源 Point-in-Time 归一。
- ADR-011：本地 8GB GPU 支持完整慢速流程。
- ADR-012：本地 core/app 常驻，重型组件按需。
- ADR-013：未来影响图候选稀疏解码。
- ADR-014：Python 是金融语义唯一实现端。

---

## 29. 进入实施前的硬门槛

开始编写代码前必须完成：

1. 用户审核并确认本设计文档。
2. 创建 `legacy-r4-before-isd-esd-rewrite` Tag。
3. 记录当前 main SHA 和旧测试结果。
4. 编写详细实施计划。
5. 生成 ZCode 总控提示词与阶段提示词。
6. 冻结第一阶段完成定义。
7. 建立提交粒度与回滚规则。

本设计通过后，下一步不是直接让 ZCode 无边界修改，而是基于本设计生成可执行、可验收、可回滚的全量实施计划。
