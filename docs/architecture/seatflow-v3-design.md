# SeatFlow V3 全新架构设计

- 状态：冻结设计
- 日期：2026-08-06
- 目标分支：`rewrite/seatflow-v3`
- 项目主线：基于席位—股票动态异构图的龙虎榜资金传播与短期收益预测
- 设计性质：从零重新设计，旧 V2 重构方案不再作为实现依据

---

## 1. 设计结论

SeatFlow V3 被定义为一个完整的 A 股龙虎榜资金研究与模拟交易系统：

```text
多源龙虎榜与行情数据
→ Point-in-Time 数据治理
→ Event / Cycle / Sample / Universe
→ Base Market Return
→ Seat / Event Direct Effect
→ Dynamic Flow Graph
→ Spillover Effect
→ t+1 / t+3 / t+5 累计市场超额收益
→ Walk-Forward / Blind / 成本后回测
→ 每日推理 / 解释 / 模拟盘
```

系统必须同时回答四个问题：

1. 哪些席位正在形成有效资金合力。
2. 这些席位对当前上榜股票产生了多大直接影响。
3. 资金是否向同行业、同概念、同席位历史关联或接力股票传播。
4. 在真实交易约束和成本下，预测是否具有可交易价值。

最终预测被严格分解为：

```text
Final Return = Base Return + Direct Effect + Spillover Effect
```

模型名称冻结为：

> **SeatFlow-FDG：Seat Flow Diffusion Graph Network**

---

## 2. 与旧方案的边界

旧仓库只保留以下价值：

- 数据源适配经验。
- Point-in-Time、防泄漏、Cycle 边界测试经验。
- 历史负结果、失败案例和数值异常记录。
- 回测、成本、模拟盘和图可视化经验。

以下内容不继承：

- 旧 `DT-SeatHGT V3` 作为主模型的地位。
- 旧 Python 包结构。
- 旧 `model_valid`、`dataset_valid`、Blind 或部署状态。
- 旧 V2 的 `apps/services/packages` 大型 Monorepo 结构。
- 旧架构中先搭建重平台、再逐渐填模型的实施顺序。
- 任何未被新测试覆盖的旧业务语义。

旧代码通过 Git Tag 保留，不整体复制到新目录。新系统不得对旧模块形成隐式依赖。

---

## 3. 产品范围

### 3.1 V3 首版必须交付

- 多数据源采集与血缘。
- 席位身份归一和历史版本。
- Event、Cycle、Sample、Universe。
- Stock、Seat、Event、Theme、Market 特征。
- t+1、t+3、t+5 累计市场超额收益标签。
- 五日日级收益路径。
- Base OOF 与 Seat Residual。
- Direct Pool 与 Spillover Pool。
- SeatFlow-FDG 模型。
- Walk-Forward、Embargo、多随机种子和密封 Blind。
- 基线、消融、Placebo 和 Shuffled Graph。
- 真实交易约束和成本后回测。
- 每日推理、传播路径解释和模型注册。
- Python Research Engine。
- Spring Boot Control Center。
- Next.js Research Console。
- 模拟盘和风险控制。
- Docker Compose、本地开发和服务器部署。
- 单元、契约、数据、模型、集成和端到端测试。

### 3.2 首版不做

- 自动真实券商下单。
- 智能体交易决策。
- 强化学习交易策略。
- 新闻、社媒、公告的大模型融合。
- 高频盘口或毫秒级交易。
- 无限多跳传播。
- 全市场 N×N 稠密邻接矩阵。
- 未经验证的因果结论。

这些扩展不得阻塞 V3 主闭环。

---

## 4. 系统架构

V3 只保留三个正式应用：

```text
SeatFlow V3
├── research-engine      Python / FastAPI / PyTorch / PyG
├── control-center       Java 21 / Spring Boot
└── research-console     Next.js / TypeScript
```

基础设施：

```text
PostgreSQL   控制面权威元数据
Redis        队列、缓存、锁和实时进度
MinIO        Dataset、图、Checkpoint、报告和解释产物
MLflow       实验追踪与模型注册
Prometheus   指标采集
Grafana      监控展示
```

数据分析和训练事实存储：

```text
Parquet + DuckDB + Manifest
```

首版不要求 Apache Iceberg、Temporal 或 Neo4j 成为硬依赖。图可视化读取已封存的图快照和路径解释产物。

---

## 5. 仓库结构

```text
SeatFlow/
├── research-engine/
│   ├── pyproject.toml
│   ├── src/seatflow/
│   │   ├── domain/
│   │   ├── ingestion/
│   │   ├── identity/
│   │   ├── dataset/
│   │   ├── features/
│   │   ├── labels/
│   │   ├── graph/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   ├── backtest/
│   │   ├── inference/
│   │   ├── tasks/
│   │   ├── api/
│   │   └── observability/
│   └── tests/
├── control-center/
│   ├── pom.xml
│   └── src/
├── research-console/
│   ├── app/
│   ├── features/
│   ├── components/
│   ├── lib/
│   └── tests/
├── contracts/
│   ├── openapi/
│   ├── json-schema/
│   └── events/
├── configs/
│   ├── dataset/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── backtest/
├── infra/
│   ├── compose/
│   ├── database/
│   ├── monitoring/
│   └── k3s/
├── tools/
│   ├── migration/
│   ├── data-audit/
│   └── release/
├── docs/
│   ├── architecture/
│   ├── data/
│   ├── model/
│   ├── api/
│   ├── testing/
│   ├── prompts/
│   └── plans/
└── legacy-reference/
    └── README.md
```

依赖规则：

- `research-console` 只调用 Control Center API。
- Control Center 不复制金融语义。
- Research Engine 是标签、Cycle、图、损失、指标、回测和模拟盘信号的唯一实现端。
- 跨语言对象由 `contracts` 统一定义。
- 生成客户端不可手工修改。

---

## 6. 核心领域对象

### 6.1 Seat

```text
seat_id
canonical_name
seat_type
identity_confidence
valid_from
valid_to
aliases
source_evidence
```

席位类型：

- BROKER_BRANCH
- INSTITUTION
- NORTHBOUND_SH
- NORTHBOUND_SZ
- SPECIAL_SEAT
- UNKNOWN

身份必须支持历史版本，禁止使用最新名称覆盖全部历史。

### 6.2 Stock

```text
stock_id
ts_code
exchange
board
listing_date
delisting_date
industry_membership_pit
concept_membership_pit
```

### 6.3 LHBEvent

一次股票在一个交易日、一个披露批次上的完整龙虎榜事件：

```text
event_id = trade_date + stable_stock_id + disclosure_batch
```

字段至少包括：

```text
stock_id
trade_date
available_time
reason_codes
total_buy
total_sell
net_amount
turnover_rate
event_confidence
source_lineage
```

### 6.4 SeatParticipation

```text
event_id
seat_id
side
rank
buy_amount
sell_amount
net_amount
amount_share
available_time
identity_confidence
```

同一 Event 中的每个买方、卖方席位保留独立事实，不压缩为单条净额边。

### 6.5 EventCycle

```text
cycle_id
stock_id
cycle_start
last_event_date
cycle_end
event_count
status
```

规则：首次上榜创建 Cycle；若新事件日期不晚于当前 `cycle_end`，归入同一 Cycle，并把 `cycle_end` 延长至新事件后第 5 个交易日；否则新建 Cycle。

### 6.6 PredictionSample

Cycle 内每个有效交易日生成滚动样本：

```text
sample_id
cycle_id
stock_id
prediction_date
cutoff_time
days_since_last_event
dataset_version
feature_schema_hash
label_policy_hash
```

### 6.7 PropagationCandidate

```text
source_event_id
source_stock_id
target_stock_id
relation_type
candidate_score
available_time
candidate_evidence
```

---

## 7. Universe 设计

### 7.1 Direct Pool

目标股票处于活跃 Cycle：

```text
0 <= days_since_last_event <= 5
```

Direct Pool 参与 Direct Loss、目标排序和直接交易回测。

### 7.2 Spillover Pool

未在当前预测日直接上榜，但通过以下关系进入候选池：

- same_seat_relay
- same_coalition_relay
- same_theme_diffusion
- same_industry_diffusion
- capital_rotation
- repeat_event_path
- historical_price_comovement_aux

价格联动只能作为辅助关系，不得成为主要传播证据。

### 7.3 Base Pool

全市场满足上市时长、流动性、交易状态和数据完整性要求的股票。Base Pool 不读取当前龙虎榜席位信息。

---

## 8. Point-in-Time 规则

每条正式记录至少区分：

```text
event_time
available_time
ingested_at
effective_from
effective_to
source_version
```

对任意 Sample：

```text
input.available_time <= sample.cutoff_time
```

席位历史收益、胜率和能力特征仅允许纳入：

```text
label_end_time <= current_cutoff_time
```

禁止：

- 未来再次上榜信息。
- 未来席位进入或退出。
- 未来停牌、涨跌停和退市状态。
- 全样本归一化。
- 同一 Cycle 跨切分。
- 训练内拟合值生成 Base Residual。
- 数值 ID 或 ID Hash 进入连续特征。

任何 Point-in-Time 证据缺失时必须 Fail-Closed。

---

## 9. 标签设计

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
ExcessReturn_h = StockReturn_h - MarketReturn_h
h ∈ {1, 3, 5}
```

同时保存五日日级路径：

```text
daily_return_1 ... daily_return_5
```

一致性必须满足：

```text
R1 = daily_return_1
R3 = Π(1 + daily_return_d), d=1..3, 再减 1
R5 = Π(1 + daily_return_d), d=1..5, 再减 1
```

辅助标签：

- direction_h
- industry_excess_h
- tail_risk_h
- tradeability_state
- relation_activation_h
- relation_direction_h
- relation_strength_h

事件研究 Mask 与可交易回测 Mask 分开保存。

---

## 10. Dataset Manifest 与 Seal

Dataset Manifest 至少包含：

```text
dataset_version
build_timestamp
date_range
source_manifest_hash
identity_policy_hash
cycle_policy_hash
feature_schema_hash
label_policy_hash
split_definition_hash
graph_schema_hash
row_counts
node_counts
edge_counts
quality_gate_results
artifact_uris
```

Dataset Seal 只有在全部硬门禁通过后生成。正式训练入口必须验证 Seal、Manifest Hash 和输入产物 Hash。

硬门禁：

- available_time 违规 = 0。
- ID/Hash 连续特征 = 0。
- 未成熟标签进入画像 = 0。
- Cycle 跨切分 = 0。
- 标签路径一致性错误 = 0。
- 关键表主键冲突 = 0。
- 图节点悬空引用 = 0。

---

## 11. SeatFlow-FDG 模型

### 11.1 Market Base Encoder

输入：

- 60 日 OHLCV。
- 换手率、流动性、波动率。
- 行业、概念和市场状态。
- 指数收益、涨跌停环境和市场宽度。
- 风险和非理性状态。

输出：

```text
base_daily_path[5]
base_return_1
base_return_3
base_return_5
market_regime_embedding
```

Base 分支不得读取当前龙虎榜席位、Event 或传播边。

### 11.2 Seat Behavior Memory

席位画像窗口：5、20、60、120 个交易日。

特征：

- 买入、卖出和净买入强度。
- 参与频率和平均排名。
- 资金集中度和方向一致性。
- 行业、概念偏好。
- 成熟历史收益和胜率。
- 接力、反转和拥挤概率。
- 身份类型和身份置信度。

冷启动席位使用 Seat Type Embedding、当前行为和低置信度掩码。

### 11.3 Event Interaction Encoder

同时建模：

- 多席位协同。
- 买卖双方冲突。
- 机构、游资和北向组合。
- 席位支配度和跟随关系。
- 净买入异常和资金结构。
- 上榜原因。
- 价格环境、事件新颖性和延续性。

输出：

```text
event_embedding
coalition_embedding
conflict_score
event_reliability
```

### 11.4 Dynamic Flow Graph

节点：

- Seat
- Event
- Stock
- Theme

Market Regime 作为全局条件，不构造全连接超级节点。

关系：

- seat_buy_event
- seat_sell_event
- event_impacts_stock
- stock_belongs_theme
- same_stock_repeat
- same_seat_relay
- same_coalition_relay
- same_theme_diffusion
- capital_rotation
- spillover_candidate

默认最多两跳传播，并对每种关系执行候选 Top-K 截断。

### 11.5 Relation State Gate

每个候选关系维护：

```text
activation_probability
impact_strength
impact_direction
reliability
age
decay
market_regime_sensitivity
```

关系状态必须直接控制：

- 边激活。
- Top-K 邻居选择。
- 消息权重。
- 注意力权重。
- 正负方向门控。

只在预测头前拼接状态向量不算完成。

### 11.6 Direct Effect Decoder

输出：

```text
direct_daily_path[5]
direct_return_1
direct_return_3
direct_return_5
continuation_probability
reversal_probability
direct_uncertainty
```

主要训练目标：

```text
SeatResidual_h = TrueExcessReturn_h - BaseOOFPrediction_h
```

### 11.7 Spillover Decoder

输出：

```text
spillover_daily_path[5]
spillover_return_1
spillover_return_3
spillover_return_5
spillover_probability
spillover_uncertainty
```

Spillover 股票在当前预测日不能是直接上榜目标。

### 11.8 Multi-Horizon Fusion

```text
FinalDailyPath = BaseDailyPath + DirectDailyPath + SpilloverDailyPath
```

日级路径再严格累计为 t+1、t+3、t+5。

最终输出对象：

```text
SeatFlowPrediction
├── stock_id
├── prediction_date
├── base_path[5]
├── direct_path[5]
├── spillover_path[5]
├── final_path[5]
├── cumulative_return_1
├── cumulative_return_3
├── cumulative_return_5
├── direction_probability_1/3/5
├── continuation_probability
├── spillover_probability
├── reversal_probability
├── uncertainty
├── top_seats
├── top_events
├── top_relations
├── top_paths
├── tradeability
└── lineage
```

---

## 12. 损失函数

```text
L = Σ_h w_h (
      λ_return L_return_h
    + λ_rank L_rank_h
    + λ_direction L_direction_h
    + λ_relation L_relation_h
    + λ_spillover L_spillover_h
    + λ_calibration L_calibration_h
    + λ_risk L_risk_h
)
```

约束：

- Return 使用 Huber 或 Quantile。
- Ranking 只在同一预测日期内比较。
- Direction 支持二分类或三分类。
- Relation 同时监督激活、方向和强度。
- Spillover 单独计算 Loss 和指标。
- Calibration 使用 Brier、ECE 和可靠性曲线。
- Risk 预测尾部风险和不可交易风险。

---

## 13. 训练流程

1. Dataset 构建和 Seal。
2. Base 模型 Walk-Forward 训练。
3. 生成严格 OOF Base 预测。
4. 构建 Seat Residual。
5. Seat Memory 与 Event Encoder 预训练。
6. Direct 模型训练。
7. Propagation Candidate 构建。
8. Relation State 和 Spillover 训练。
9. SeatFlow-FDG 联合训练。
10. Walk-Forward 评估。
11. 多随机种子。
12. 消融和 Placebo。
13. Blind Candidate Freeze。
14. 人工审批。
15. Blind Evaluation。
16. Model Registry 和 Promotion。

不得在 Dataset 未 Seal 时启动正式训练。

---

## 14. 切分与 Blind

Walk-Forward：

- Train：过去全部成熟 Cycle。
- Validation：后续约 60 个交易日。
- Embargo：至少 5 个交易日。
- Test：后续约 20 个交易日。

同一 `cycle_id` 完整进入同一个集合。

Blind：

- 最后连续约 6 个月。
- 开发期间不可读取 Blind 标签指标。
- Dataset Manifest、Git SHA、Config、Seed、Checkpoint、Cost Config 和 Blind Date Range 必须冻结。
- Blind 结果不可覆盖。
- 根据 Blind 结果继续调参后，不得仍称为同一次 Blind。

---

## 15. 模型有效性门禁

### 15.1 数据门禁

```text
dataset_valid = true
leakage_count = 0
```

### 15.2 Direct 门禁

- Real Seat > Amount Only。
- Real Seat > Seat Type。
- Real Seat > Anonymous Seat。
- Real Seat > Shuffled Seat。
- Base + Direct > Base。

### 15.3 Spillover 门禁

- Real Graph > No Graph。
- Real Graph > Shuffled Graph。
- Edge AUPR > Random。
- Spillover RankIC 在多数 Fold 和 Seed 为正。

### 15.4 经济门禁

- 成本后 Top-K 超额收益 > 0。
- 多数 Fold 为正。
- 多数 Seed 为正。
- 最大回撤和换手可解释。
- 结果不依赖单一市场阶段。

### 15.5 校准门禁

- Brier Score。
- ECE。
- Reliability Curve。
- Uncertainty Coverage。

任一关键门禁失败：

```text
model_valid = false
promotion_allowed = false
paper_trading_allowed = false
```

---

## 16. 回测与模拟盘

交易约束必须覆盖：

- t+1 开盘入场。
- 停牌。
- 一字涨停无法买入。
- 一字跌停无法卖出。
- 涨跌停价格限制。
- 退市和特殊处理。
- 滑点、佣金、印花税和冲击成本。
- Top-K 组合、持有期重叠和资金占用。
- 订单失败、部分成交和撤单。

回测结果必须同时展示：

- 成本前。
- 成本后。
- Direct Pool。
- Spillover Pool。
- 全组合。
- 市场状态分组。
- 席位类型分组。
- 上榜原因分组。

模拟盘只能使用已 Promotion 的模型，并记录模型版本、Dataset 版本、预测时间、订单、成交、失败原因和审计信息。

---

## 17. Research Engine 职责

负责：

- 数据采集和校验。
- 身份归一。
- Dataset 构建。
- PIT 查询。
- 特征和标签。
- 图和样本张量化。
- 模型训练和评估。
- 回测和模拟信号。
- 每日推理和解释。
- Worker 任务执行。
- Research API。

不得负责用户、权限和平台审批。

---

## 18. Control Center 职责

模块：

```text
auth
user
role
permission
project
data-source
dataset
quality-gate
experiment
training-run
evaluation
model-registry
promotion
prediction
backtest
paper-trading
workflow
notification
audit
system-monitor
```

异步任务流程：

```text
Java 创建任务记录
→ Redis Queue
→ Python Worker 执行
→ PostgreSQL 更新权威状态
→ Redis Pub/Sub 发布进度
→ Java SSE 转发前端
```

PostgreSQL 是任务和审批状态的权威来源；Redis 不得成为唯一状态来源。

---

## 19. Research Console 页面

至少包含：

1. 研究总览。
2. 数据源管理。
3. Dataset Builder。
4. 数据质量中心。
5. Event/Cycle/Sample 浏览器。
6. 席位身份审核。
7. 席位画像。
8. 图快照和传播图。
9. 实验创建。
10. 训练监控。
11. 模型比较。
12. 预测中心。
13. 单股票解释。
14. 回测中心。
15. 成本归因。
16. 模拟盘。
17. 模型注册与审批。
18. 系统设置、权限、审计和运行监控。

每个研究结果页面必须显示：

- 数据来源。
- cutoff_time。
- Dataset Version。
- Model Version。
- 是否正式运行。
- 是否 Blind。
- 是否扣除成本。
- 是否可交易。
- 有效性状态和失败原因。

禁止使用合成结果伪装正式结果。

---

## 20. 通信契约

### Frontend → Java

- REST + OpenAPI。
- SSE 实时进度。

### Java → Python

- REST：任务提交、查询和取消。
- Redis Queue：异步任务。
- Artifact URI：大型产物引用。

大型产物只传：

```json
{
  "artifactUri": "s3://seatflow/...",
  "manifestHash": "sha256:...",
  "schemaVersion": "seatflow.v3"
}
```

统一错误模型：

```json
{
  "code": "DATASET_QUALITY_GATE_FAILED",
  "message": "Cycle cross-split violations detected",
  "traceId": "...",
  "taskId": "...",
  "details": {},
  "retryable": false
}
```

---

## 21. 本地与服务器运行模式

Profiles：

- `local-smoke`
- `local-full-slow`
- `server-full`

`local-full-slow` 保留完整功能，通过串行、分块和断点恢复运行，不允许静默减少 Fold、Seed、时间范围、Blind 或消融项。

本地资源策略：

- GPU Worker 并发 1。
- Fold 串行。
- Seed 串行。
- AMP。
- Gradient Accumulation。
- Gradient Checkpointing。
- CPU 保存图、按批搬运 GPU。
- Windows DataLoader Worker 默认 0–2。
- OOM 后禁止静默切 CPU。

---

## 22. 测试策略

### Python

- Domain 单元测试。
- PIT 泄漏测试。
- Cycle 边界和延长测试。
- 标签公式和路径一致性测试。
- OOF 独立性测试。
- ID 禁入连续特征测试。
- 图 Schema 和候选 Top-K 测试。
- Shape、Gradient、Micro-overfit 测试。
- Walk-Forward 原子性测试。
- Blind 访问测试。
- 回测执行规则测试。

### Java

- Controller、Service、Repository 测试。
- RBAC。
- 任务幂等。
- Redis Queue。
- SSE 重连。
- 审批和审计门禁。

### Frontend

- 组件和 Adapter 测试。
- 空态、错误态、降级态和无效模型状态。
- SSE 重连。
- 权限显示。
- Playwright E2E。

### 端到端

合成数据必须跑通：

```text
ingest
→ identity
→ cycles
→ features
→ labels
→ graph
→ samples
→ quality seal
→ base OOF
→ direct smoke
→ spillover smoke
→ evaluation
→ prediction
→ Java API
→ Next.js display
```

---

## 23. 可观测性与安全

统一关联：

```text
trace_id
task_id
dataset_version
experiment_run_id
model_version
prediction_date
```

必须监控：

- 数据源延迟。
- 分区构建进度。
- 数据质量指标。
- GPU 显存和利用率。
- Fold/Seed 进度。
- API 延迟。
- Worker 队列长度。
- SSE 连接状态。
- MinIO 和 PostgreSQL 健康状态。

安全要求：

- 数据源密钥通过 Secret 注入。
- 前端不获取真实 Token。
- Blind 标签访问单独授权和审计。
- Dataset Seal、Model Promotion 和模拟盘启用必须有权限控制。
- Git 中禁止出现真实密钥。

---

## 24. 全量开发轨道

```text
Track A：仓库与契约
Track B：数据集
Track C：模型与训练
Track D：Python API 与 Worker
Track E：Java Control Center
Track F：Next.js Console
Track G：基础设施、测试与发布
```

所有轨道从第一天并行开发，但依赖统一契约。任何页面和 API 可以提前实现空态、状态机和契约，不得伪造正式研究结果。

---

## 25. 验收标准

### 数据

- 任意 Sample 可追溯到源记录。
- 任意 Event 可恢复全部 SeatParticipation。
- 任意 Cycle 可恢复全部 Event 和每日 Sample。
- 所有正式输入满足 available_time。
- 标签和日级路径严格一致。
- 同一 Cycle 不跨 Split。
- Base OOF 和 Seat Residual 有独立血缘。

### 模型

- Base、Direct、Spillover 均有独立输出和指标。
- Relation State 直接控制消息路由。
- Spillover 股票不与 Direct Pool 混淆。
- Real Seat、Shuffled Seat、No Graph、Shuffled Graph 消融齐全。
- 8GB 模式可完成前向、反向、Checkpoint 和恢复。

### 平台

- Java 可创建、查询、取消和恢复任务。
- 前端刷新后可恢复任务状态。
- Quality Gate 失败时训练和模拟盘均被阻断。
- 模型无有效证据时不展示正式预测。
- Trace ID 可串联数据、训练、评估、预测和订单。

### 部署

- Docker Compose 可启动核心服务。
- CI 覆盖 Python、Java、Frontend 和契约。
- 同一镜像可用于本地和服务器。
- Secret 不进入仓库。

---

## 26. 架构决策记录

- ADR-001：V3 从零设计，旧 V2 不作为实现底座。
- ADR-002：三个正式应用，避免首版微服务膨胀。
- ADR-003：Parquet + DuckDB + MinIO 作为研究数据主路径。
- ADR-004：PostgreSQL 保存控制面权威状态。
- ADR-005：Redis 用于队列、缓存和实时进度，不作为唯一状态源。
- ADR-006：Python 是金融研究语义唯一实现端。
- ADR-007：Base + Direct + Spillover 显式分解。
- ADR-008：Cycle 是数据切分原子。
- ADR-009：Relation State 必须进入消息路由。
- ADR-010：Spillover 独立门禁决定能否声称资金传播有效。
- ADR-011：本地 8GB GPU 支持完整慢速流程。
- ADR-012：全量范围一次冻结，多轨并行、逐门禁联调。

---

## 27. 实施入口

实施必须以以下文档为准：

- `docs/architecture/seatflow-v3-design.md`
- `docs/plans/seatflow-v3-full-implementation-plan.md`
- `docs/prompts/zcode-master-prompt.md`
- `docs/prompts/zcode-phase-00-to-15.md`
- `docs/testing/seatflow-v3-test-matrix.md`

旧 V2 文档保留为历史记录，但不得被 ZCode、开发者或 CI 当作 V3 规范来源。