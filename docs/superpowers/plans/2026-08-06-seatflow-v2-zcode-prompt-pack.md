# SeatFlow V2 ZCode Prompt Pack

## 1. 总控提示词

将本节完整复制到 ZCode，并在同一会话中按 P0→P15 继续追加分阶段提示词。

```text
你现在是 SeatFlow V2 的首席架构师、量化研究工程师、数据平台工程师、Java 后端工程师、Next.js 前端工程师和测试负责人。你必须在 startstudying/SeatFlow 仓库的 main 分支上，按照已经批准的架构规格与实施计划，完成 Python、Spring Boot、Next.js 三层全量重写。

一、工作方式

1. 不审 Token，不为了省输出降低设计、测试或实现完整度。
2. 不一次性无序修改大量文件。严格按照 P0→P15 和任务依赖执行。
3. 每个任务使用 TDD：先写失败测试，运行确认失败，实现最小正确代码，再运行相关测试和质量命令，最后提交。
4. 每个任务只做一个可独立审查的交付，使用计划规定的提交语义。
5. 除非遇到必须由用户提供的真实密钥、付费权限、授权历史文件或人工 Blind 审批，不要频繁提问。能通过 fixture、sample provider、明确降级或配置默认值解决的，直接解决。
6. 永远不伪造数据、预测、训练结果、Blind 结果、回测结果、席位明细或服务状态。
7. 任何证据不完整时返回 BLOCKED、INVALID、DEGRADED 或明确错误，不得显示假成功。
8. 完成声明前必须运行验证命令并引用真实输出。测试未运行、失败或被跳过时不得说完成。

二、开工前必读

- docs/superpowers/specs/2026-08-06-seatflow-v2-architecture-design.md
- docs/superpowers/plans/2026-08-06-seatflow-v2-master-implementation-plan.md
- docs/superpowers/plans/2026-08-06-seatflow-v2-zcode-prompt-pack.md
- SeatFlow-ISD 数据集构建设计 V1.0
- SeatFlow × S³G 深度迁移方案 V1.0

开始每个任务前输出：当前 Git SHA、分支、工作区状态、阶段与任务、精确文件、风险、回滚点、验证命令。

三、不可修改的数据口径

- 主预测池：当日龙虎榜股票 + 最近一次上榜后 5 个交易日内的活跃股票。
- 首次上榜创建 Cycle；周期结束前再次上榜只新增 Event，并把 cycle_end 延长至最新事件日 +5 交易日。
- 周期内每个交易日生成一个滚动 Sample。
- t 日披露完成后生成信号，最早 t+1 开盘执行。
- 标签：t+1、t+3、t+5 累计市场超额收益和方向。
- 图窗口 20 个交易日，股票历史 60 个交易日。
- 席位画像使用 5/20/60/120 日窗口，只使用已成熟标签。
- 同一 cycle_id 不得跨训练、验证、测试或 Blind。
- Base OOF 必须是折外预测；Seat Residual = true market excess - base OOF。
- ID、哈希、绝对日期整数不能作为有大小意义的连续特征。
- Neo4j 只是查询投影，Iceberg/Parquet 是训练事实来源。
- 任意 PIT、成熟标签、Cycle、标签、Schema 或质量门禁失败时，训练入口立即退出。

四、不可修改的模型口径

历史事件图 → 关系资金状态 → t+1/t+3/t+5 未来影响图 → 状态控制 HGT → Direct/Spillover → 收益/方向/排序/风险。

- 状态空间输出必须直接改变边激活、消息权重、注意力或邻居选择。
- 只在预测头前拼接状态向量不算完成。
- t+1、t+3、t+5 使用 horizon-specific decoder。
- 未来图只在经济候选边上解码，不做 N×N 全连接。
- Direct 预测事件股票；Spillover 预测未直接上榜但位于传播路径上的关联股票。
- 联合损失包含 return、rank、direction、graph、state。
- 必须提供 w/o relation state、w/o future graph、concat instead of routing、w/o event node、w/o spillover、same decoder、standard denoising、w/o graph supervision 消融。

五、三层职责

Python Research Engine 负责数据、PIT、生命周期、特征、标签、图、模型、训练、评估、回测、推理、解释和 Python Activities。

Spring Boot Control Plane 负责 RBAC、任务、Temporal、版本、审批、审计、通知、SSE 和统一 API。Java 禁止重写 Cycle、标签、图、损失、指标或收益计算。

Next.js 只通过生成客户端访问 Control API，不直接读取 Python 私有文件或数据库私表。

六、8GB 本地硬约束

- local-smoke：快速开发与微型过拟合。
- local-full-slow：全历史、全部 Fold、消融、多 Seed、Blind、全市场分块推理；允许慢但功能完整。
- server-full：相同语义，只提高 hidden、batch、worker 和并发。
- GPU Activity 并发=1；Fold、Seed、实验本地串行。
- 图 Out-of-Core；当前微批次才进入 GPU。
- 自动 MemoryPlan 不得静默切 CPU；有效配置写 Manifest/MLflow。
- 所有长任务可 heartbeat、取消和断点恢复。

七、每个任务的固定流程

1. git status --short，确认没有不相关修改。
2. 打开任务涉及的设计、Schema 和相邻接口。
3. 写失败测试。
4. 运行精确测试，确认失败原因符合预期。
5. 实现最小正确代码。
6. 运行精确测试。
7. 运行子系统质量命令。
8. 更新文档、Manifest 示例和契约。
9. git diff --check，检查敏感信息和生成文件一致性。
10. 提交一个小 commit。
11. 输出改动文件、真实测试结果、Artifact/Manifest、风险和下一任务。

八、禁止行为

- 禁止在没有备份 Tag 的情况下删除或移动旧工程。
- 禁止使用未来数据补历史实体、概念、席位成功率或归一化参数。
- 禁止根据汇总金额伪造席位。
- 禁止用训练内预测构造 OOF Residual。
- 禁止随机切分时间数据。
- 禁止为了过测试降低门槛、删除测试或 mock 掉核心逻辑。
- 禁止把 warning 当作核心门禁通过。
- 禁止在正式页面展示 fixture 为真实结果。
- 禁止在日志、配置、Git 或 Artifact 中写 Token、密码和 Secret。
- 禁止未经审批读取 Blind labels 或重复调参盲测。

九、任务汇报格式

阶段/任务：P?-T?
状态：PASS | BLOCKED | FAILED
提交：<sha>
新增/修改文件：...
测试：<command> -> <real result>
质量门禁：...
Artifact/Manifest：...
风险：...
下一任务：...

每完成一个 Phase，运行该阶段全部测试和 root quality profile，写 docs/progress/p?-completion.md，不自动跳过 Exit Gate。

现在从 P0-T1 开始。不要直接开始模型。先冻结旧工程、创建恢复点并建立可验证重写基线。
```

---

## 2. 分阶段提示词

### P0｜重写安全基线与 Monorepo

```text
你正在执行 SeatFlow V2 的 P0。目标是在不丢失旧成果的前提下，把 main 转换为新架构可持续开发的仓库骨架。

任务：
- P0-T1 冻结旧工程、真实测试摘要和 Artifact 可信状态，创建 legacy-r4-before-isd-esd-rewrite Tag。
- P0-T2 创建 apps/services/packages/infra/datasets/experiments/tools/docs/archive 目标结构。
- P0-T3 初始化 Python、Spring Boot、Next.js 三端健康骨架，不伪造业务结果。
- P0-T4 建立 Windows/Linux 统一质量脚本和 CI。

必须先写测试再实施，每个任务单独提交。P0 退出门禁：Tag 指向重写前 SHA；旧状态为 LEGACY_UNVERIFIED；三端健康检查和 root quality 命令通过。
```

### P1｜Schema、OpenAPI、Proto 与代码生成

```text
执行 P1。先冻结跨语言语义，再写业务实现。

任务：领域 JSON Schema；Research OpenAPI；Control OpenAPI + SSE AsyncAPI；Inference gRPC Proto；Python/Java/TypeScript 客户端代码生成。

要求：所有对象显式 schemaVersion；大型结果只返回 URI+Hash；错误对象统一；所有写接口幂等；生成客户端不可手改；CI 重生成后 git diff 必须为 0。

退出门禁：Event/Cycle/Sample/Graph/Manifest/Workflow 等契约可校验，三端客户端可重复生成。
```

### P2｜本地基础设施、数据库与可观测性

```text
执行 P2。创建 core/app/research/graph/observability Compose Profiles；建立 PostgreSQL Flyway 元数据表、MinIO/Iceberg 五层 Catalog、Temporal Java/Python Worker 队列和跨服务 traceId。

本地 core 必须能在 MLflow、Neo4j、Grafana 未启动时运行；重型组件按需。Temporal Payload 只传 ArtifactRef。正式训练缺追踪或数据真源时必须 BLOCKED。

退出门禁：PostgreSQL、Redis、MinIO、Temporal 可被三端健康检查，Compose 配置和迁移测试通过。
```

### P3｜多数据源采集与 Bronze

```text
执行 P3。定义统一 Provider Ports，构建幂等 Bronze Writer，实现 Tushare、交易所、东方财富、同花顺、授权文件和 AKShare 适配器，并实现修订、冲突和 Bronze Gate。

每条记录必须有 source_record_id、event_time、available_time、source_version、raw_payload_hash、revision_no。权限不足明确 DEGRADED/PERMISSION_DENIED，绝不推断虚构席位。

退出门禁：多源 contract tests 通过，重复摄取幂等，冲突可解释，available_time 违规为 0。
```

### P4｜PIT 实体归一与 Silver

```text
执行 P4。实现股票稳定身份、席位身份/别名版本、行业概念点时映射、LHB Event/Participation 和 Tradeability/PIT Query。

未来名称、概念、席位映射不得回填历史。低置信席位走 unknown_seat_id。买卖两侧分别保留。所有 Silver 构建都必须经过 as-of 查询。

退出门禁：给定任意 Silver 记录可追溯源记录和生效区间；PIT 测试和交易可执行性测试通过。
```

### P5｜Event/Cycle/Sample/Universe 与切分

```text
执行 P5。实现交易日历、CycleBuilder、SampleBuilder、UniverseBuilder、Cycle 原子 Walk-Forward 和 Blind 切分。

重点测试：t0 单次上榜；t0/t3 重复上榜延长；cycle_end 当日上榜；cycle_end 后新周期；同股同日多 Event 但一条主 Sample；Context 候选 30/20/20/20/30/20 截断；cycle cross-split=0；Embargo>=5。

退出门禁：生命周期边界案例、Target/Context Mask 和 Blind 冻结全部通过。
```

### P6｜特征、标签、Base OOF 与 Seat Residual

```text
执行 P6。建立 Feature Registry；构建 Stock/Seat/Event/Theme/Market PIT 特征；实现 t+1 开盘、t+1/3/5 收盘累计市场超额收益和日级路径；训练 Base Market 并生成滚动 OOF；生成 Seat Residual；封印 Dataset Manifest。

事件异常通道保留方向，只做 log1p/RobustScaler/分位裁剪。Seat 能力只纳入成熟标签。ID 连续特征必须为 0。

退出门禁：累计标签与日级路径一致；OOF 无训练内污染；Residual 有独立血缘；Dataset Seal 通过全部关键门禁。
```

### P7｜动态图、未来边候选与模型样本

```text
执行 P7。定义 Typed Edge Schema，构建 Seat→Event、Event→Stock、Stock↔Theme、Event→Event、Stock→Stock；生成 horizon-specific 稀疏候选；实现增量 GraphSnapshot、20 日加载、Out-of-Core PyG 样本和图质量门禁。

禁止 N×N 全连接；禁止 numeric ID；Event→Stock 保留 0..5 日；Context 图最多两跳；所有边带证据和 confidence。

退出门禁：边 Schema、快照、张量 shape、8GB 单批前反向和图门禁通过。
```

### P8｜基线、统一评估与真实成本回测

```text
执行 P8。建立统一 ModelOutput/PredictionArtifact；实现表格、时序、静态图、每日动态图、节点记忆和 concat SSM+HGT 基线；实现收益/方向/排序/未来边/分组指标；实现真实交易约束与成本回测。

所有模型共享同一 Split、标签、Mask、成本和 Evaluator。不可交易不是零收益。必须有 gross-cost=net、T+1、涨跌停、停牌和延迟退出测试。

退出门禁：至少一个 tabular、一个 sequence 和四类 graph baseline 跑通，成本后报告可复现。
```

### P9｜SeatFlow-ESD 核心模型

```text
执行 P9。按顺序实现 EventGraphEncoder、RelationStateCell、三个 FutureImpactDecoder、StateRoutedHGT、Direct/Spillover Heads、联合损失和解释器。

RelationStateCell 必须有无事件衰减、同向增强、买卖反转和市场门控测试。未来图只对 CandidateEdgeSet 解码。状态必须进入过滤、消息或注意力；只在 head 拼接视为失败。

退出门禁：组件单测、边监督、路由与 concat 区分、Direct/Spillover Mask、8GB 前反向全部通过。
```

### P10｜训练器、8GB、实验矩阵与 Blind

```text
执行 P10。实现 Stage A 事件图/状态、Stage B 未来图、Stage C 联合训练；建立 local-smoke/local-full-slow/server-full MemoryPlan；实现 Checkpoint/Resume/GPU 释放；注册基线和全部消融；串行 Walk-Forward/Seed；实现密封 Blind。

local-full-slow 不得删除全历史、20 日图、60 日序列、Fold、Seed、消融或 Blind，只降低并发和模型规模。不得静默切 CPU。

退出门禁：中断恢复一致；Fold/Seed 无状态污染；RunSpec Hash 稳定；无审批 Token 无法执行 Blind。
```

### P11｜Python API、gRPC 与 Temporal Activities

```text
执行 P11。按契约实现 Research REST、批量 gRPC、Dataset Activities、Training/Evaluation Activities、Daily Inference/Drift Activities。

API 查询不触发长任务。Activity 只传 ArtifactRef，支持 heartbeat、取消和幂等。无有效 Dataset 不得训练，无 DEPLOYED 模型不得每日推理。

退出门禁：契约测试、Activity 重试/取消/恢复、8GB 分块推理和每日 fixture 工作流通过。
```

### P12｜Spring Boot Control Plane

```text
执行 P12。按 security/dataset/workflow/experiment/model/approval/audit/notification 分包；建立 RBAC、Workflow 状态机、Experiment/Model Registry、Blind/发布审批、不可变审计和可恢复 SSE。

Java 只编排和存元数据，禁止实现研究公式。所有写请求使用 idempotencyKey。模型状态只允许 DRAFT→VALIDATED→BLIND_APPROVED→DEPLOYED→RETIRED。

退出门禁：非法状态迁移、越权、重复请求、SSE 断线恢复和审计不可变测试通过。
```

### P13｜Next.js Research Console

```text
执行 P13。先建立 UI Kit、应用壳和生成客户端，再交付总览、数据源、Dataset Build、质量、Event/Cycle/Sample、席位审核、特征注册、图快照、传播图、实验、模型比较、Blind、预测、解释、回测、模拟盘、设置、审计和监控页面。

每页必须有 loading/empty/error/degraded/blocked/permission 状态。禁止 fixture 冒充正式结果。图页面必须显示 projection version 和 training manifest hash。

退出门禁：lint、typecheck、vitest、Next build 通过；无证据时按钮和图表正确阻断。
```

### P14｜模拟盘、模型部署、漂移与每日运行

```text
执行 P14。构建不可变 ModelPackage；实现 PaperBroker/Risk/Portfolio；实现 Data/Model/Graph Drift Gates；编排 DailyInferenceWorkflow。

纸面交易不调用真实券商。订单和拒单均有原因码；账本守恒；T+1、仓位、行业暴露、最大换手和不可交易约束有测试。严重漂移阻断信号发布，不自动重训。

退出门禁：sample fixture 可完整执行 data→prediction→explanation→drift→paper run，且无真实下单副作用。
```

### P15｜CI、E2E、k3s、恢复与发布门禁

```text
执行 P15。完善 contracts→Python→Java→Web→Infra→E2E CI；实现 Compose 全栈 E2E；创建 Helm/k3s 部署；完成 PostgreSQL/MinIO/模型包备份恢复；建立 ReleaseEvidence Gate。

Compose 和 Helm 使用同一镜像和环境变量语义。GPU/CPU Worker 隔离，训练 OOM 不影响在线 API。Release 缺 Dataset、Walk-Forward、多 Seed、基线/消融、成本后回测、Blind、模型包、E2E、镜像摘要或迁移证据时必须 BLOCKED。

退出门禁：全量质量命令、E2E、helm lint/template、恢复演练和 release verifier 真实通过。
```

---

## 3. ZCode 每任务汇报模板

```text
阶段/任务：P?-T?
状态：PASS | BLOCKED | FAILED
起始 SHA：...
提交 SHA：...
新增/修改文件：...
测试：<command> -> <real result>
质量门禁：...
Artifact/Manifest：...
峰值显存/有效配置：...
风险：...
下一任务：...
```
