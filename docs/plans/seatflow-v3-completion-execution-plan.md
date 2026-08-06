# SeatFlow V3 Completion Execution Plan

**Goal:** 将 `rewrite/seatflow-v3` 从“已推送的部分实现”推进到可复现、fail-closed、研究证据可审计的 SeatFlow-ISD 交付；工程可运行与模型有效性必须分别判定。

## Global Constraints

- 所有正式输入必须满足 `available_time <= sample.cutoff_time`，任一违规都必须阻断 Dataset Seal。
- 股票、席位、事件、关系 ID 及其 hash 只能用于索引、键或离散 embedding lookup，不得进入连续特征、候选评分、边权或 MLP 输入。
- Base 必须不读取当前 Seat/Event；Base residual 必须来自严格 expanding walk-forward OOF 预测。
- Direct 只拟合 `y_true_daily_path - stop_gradient(y_base_oof_daily_path)`；Diffusion 只拟合 Direct 之后的剩余传播残差。
- 正式预测目标是未来 1–5 个交易日的日级路径；t+1/t+3/t+5 必须由同一路径累计得到，不能由三个互不相关的 head 直接生成。
- Real Seat、Seat Type、Anonymous、Seat Shuffle、Graph Shuffle、No Graph 必须共享冻结数据、切分、seed 和成本配置。
- 未完成真实数据 lineage、多折多 seed、Placebo、sealed blind once、成本后回测前，`model_valid`、`backtest_valid`、`competition_ready` 必须保持 `false`。
- API、Worker、Java 和 UI 不得把样例、合成或固定数值包装成正式研究结果；缺少真实 artifact 时必须返回明确的阻断状态。
- 不添加新的第三方依赖；确有必要时先停在任务报告中说明。
- 所有行为改动执行 TDD：先运行聚焦测试取得预期 RED，再写最小实现并取得 GREEN。
- 每个任务由独立审查者进行规格与质量审查；Critical/Important 问题修复后才能进入依赖任务。

### Task 1: 修复 V3 CI 与前端可复现安装

**Ownership:** `.github/workflows/v3-ci.yml`、`research-console/package.json`、`research-console/package-lock.json`，不得修改 Python 业务代码。

**Work:**

- 修正 V3 CI 中 Python/contracts/infra 测试依赖，使每个 job 在干净 runner 上具备 `pytest`。
- 修正 React 19 与测试工具 peer 依赖冲突，生成并提交确定性的 npm lockfile。
- 保留 lint、typecheck、test、build 四个前端门；禁止使用 `--legacy-peer-deps` 掩盖冲突。

**Acceptance:**

- `npm --prefix research-console ci` 成功。
- `npm --prefix research-console run lint`、`typecheck`、`test`、`build` 全部成功。
- workflow 语法有效，contracts 与 infra job 可从声明的安装步骤获得 pytest。

### Task 2: 修复 Python 确定性、release gate 与静态质量门

**Ownership:** `research-engine/pyproject.toml`、`research-engine/src/seatflow/ingestion/manifest.py`、`research-engine/src/seatflow/ingestion/models.py`、`research-engine/src/seatflow/release/`、对应测试及 Task 3 之外的 Ruff 机械修复。

**Work:**

- 将错误的 `python-jsonschema` 发行包名改为正确依赖。
- 新增 manifest hash 回归测试，证明相同业务输入在不同构建时刻保持同 hash，同时保留运行时间字段的可审计性。
- 新增 release gate 回归测试，证明扫描器不因自身 forbidden-term 常量误报，但仍能阻断真正的正式声明。
- 使 `python -m ruff check research-engine` 归零；不得用全局 ignore 隐藏错误。

**Acceptance:**

- 聚焦测试记录 RED/GREEN，`research-engine` 全量测试通过。
- `python -m ruff check research-engine` 通过。
- release gate 对当前 fail-closed 仓库返回与真实状态一致的结构化结果，不再自扫描误报。

### Task 3: 移除 ID/hash 连续特征与伪候选评分

**Ownership:** `research-engine/src/seatflow/graph/tensorize.py`、`research-engine/src/seatflow/graph/candidates.py`、相关 schema/validator 与对应测试。

**Work:**

- 先新增回归测试，证明改变 node/stock ID 不会改变连续 tensor、edge attributes 或传播候选分数。
- tensorization 只读取显式语义连续特征；缺失、非有限值、维数漂移和被禁 identity 字段必须 fail-closed。
- 候选评分改为由 cutoff 前的关系证据和语义特征决定；证据不足时拒绝生成，不得由对象顺序或 Python hash 决定。

**Acceptance:**

- ID 重命名不改变连续数值输出。
- 禁用 identity 字段进入连续特征的 validator 测试通过。
- 同输入跨 Python 进程/哈希随机种子产生一致候选排序。

### Task 4: 统一 canonical V3 runtime 并清除模拟正式结果

**Ownership:** V3 Docker/Compose 入口、Research API/Worker 状态适配、Java gateway/controller、Console API adapter 及其契约测试。

**Parallel slices:**

- **4A Python runtime:** `research-engine/src/seatflow/api/`、`worker/` 及其测试；默认 store 为空，默认 handler 在真实 pipeline adapter 缺失时进入 `BLOCKED`。
- **4B Control/UI boundary:** `control-center` controller/gateway 与 `research-console/lib/api` 及其测试；控制面不生成研究数值，只转发或返回机器可读阻断。
- **4C Deployment:** root/V3 Dockerfile、`deployment/`、`infra/compose/` 与部署测试；唯一研究进程为 `uvicorn seatflow.api.app:app`，health path 为 `/research/v1/health`。

三条 slice 不得修改彼此所有权文件；合并后作为一个 Task 4 checkpoint 统一审查。

**Work:**

- 冻结 `research-engine` 的 `/research/v1` 为研究语义唯一实现端；Java 只做控制面与治理。
- Root/deployment runtime 不再启动 legacy `src/seatflow` 样例 API。
- Worker 调用真实 pipeline adapter；pipeline/artifact 缺失时返回 `BLOCKED` 和机器可读原因。
- 删除固定 accuracy/Sharpe/prediction 与预置“正式”状态；允许明确标记的 `SYNTHETIC_NON_FORMAL` 测试夹具。

**Acceptance:**

- 契约测试覆盖 task 状态、artifact lineage、阻断原因和禁止正式 mock 声明。
- Compose config 与镜像入口统一指向 V3 应用。
- Java/Console 对缺失研究 artifact 展示 fail-closed，而非 0.0 或样例指标。

### Task 5: 完成 P0 数据审计与 P1 Dataset v3 materializer

**Ownership:** ingestion、identity、dataset、feature schema、seal、lineage 和数据审计 CLI。

**Parallel slices:**

- **5A Computed gates:** Dataset materializer、seal、sample active window 与 feature/topology validators；调用方不得自行传入 gate 布尔值。
- **5B Lineage/identity:** Bronze→Silver→Gold→Tensor artifact lineage、PIT identity coverage、供应商字段编码与低置信降级证据。
- **5C Real-data audit:** 只读审计 CLI/报告，统计数据窗口、字段覆盖、available-time 违规、标签成熟率、identity confidence、disposition 与 fingerprints；数据不足时返回 blocker。

三条 slice 只共享冻结的 gate/report DTO；不得以合成 fixture 生成正式 P0 通过报告。

**Work:**

- 建立 Bronze/Silver/Gold/Tensor 可追溯链；修复席位/供应商字段编码。
- 自动计算 available-time、label maturity、identity coverage、disposition、schema/hash、topology、ID-feature 等质量门。
- 用交易日历硬编码 5 日 active window；支持图节点不得进入主评价池。
- Dataset Seal 只接收由 materializer 生成的门禁证据，不接收调用方随意传入布尔值。

**Acceptance:**

- 单样本可从 Tensor 追溯到源记录与可用时间。
- 任一 PIT、标签成熟、schema、identity/hash 连续特征违规都阻断 Seal。
- 生成真实数据覆盖报告；数据不可得时输出明确 blocker，不以合成数据替代。

### Task 6: 完成 P2 Base 日级路径与严格 OOF

**Ownership:** Base 模型、walk-forward splitter、OOF artifact、训练 CLI 与测试。

**Work:**

- 输出未来 1–5 个交易日日级路径，并由路径累计 t+1/t+3/t+5。
- expanding walk-forward 产生多个 folds，包含 purge/embargo 与 fold-local transforms。
- Base 输入审计禁止 Seat/Event/当前事件派生特征；OOF manifest 记录 fold、cutoff、dataset/config/git hash。

**Acceptance:**

- 至少 3 个合成协议 fold 的边界/泄漏测试通过；真实数据 fold 数由冻结配置决定。
- 每个训练样本只使用其 OOF Base prediction 作为后续 residual 基线。
- no-seat 反事实下最终预测精确退化为 Base。

### Task 7: 完成 P3 Direct residual 与消融晋级门

**Ownership:** Seat/Event encoder、Direct head、残差 loss、ablation runner、model gate。

**Work:**

- Direct 只拟合 Base OOF residual；Base 参数默认冻结。
- 实现 Real Seat、Type、Anonymous、Seat Shuffle 的同协议消融。
- 低置信席位退回 type/unknown prior，不输出高确定性解释。

**Acceptance:**

- 测试证明 total target 不能误传给 Direct loss。
- 未具备完整多 fold/seed 消融时 gate 保持 `model_valid=false`。
- Real Seat 未稳定优于对照时记录负结果并阻断 Diffusion promotion。

### Task 8: 完成 P4 Diffusion 与 P5 独立适配实验

**Ownership:** relation state、relay candidates、diffusion head、state/sparse/placebo loss、adaptation runners。

**Work:**

- 候选边只来自 cutoff 前合法证据；最多两跳、训练折内调 Top-K。
- Relation State 直接控制激活、邻居选择、消息和注意力。
- Diffusion 拟合 Direct 后残差；实现 Graph Shuffle、No Graph 和反事实。
- Proceed/DynaME/ADAPT-Z/DoubleAdapt 逐项独立实验，未通过不得合入最终配置。

**Acceptance:**

- 关系状态变化可观测地改变 routing，ID 不影响 routing。
- 真实图未稳定优于打乱图时保持 Direct-only 降级。
- 适配模块各有独立 artifact、指标和 promotion decision。

### Task 9: 完成 P6 sealed blind、成本后回测与 P7 服务绑定

**Ownership:** candidate freeze、blind evaluator、cost-aware backtest、promotion、daily inference、真实 artifact UI/API。

**Work:**

- 冻结 dataset、schema、config、seed、checkpoint、cost、git SHA 后生成 candidate manifest。
- sealed blind 对同一 candidate 只执行一次，报告不可覆盖。
- 回测使用 T+1 可成交价、停牌/涨跌停 disposition、成本和滑点。
- 服务仅加载 promoted immutable model，并返回完整 lineage；未 promotion 时保持阻断。

**Acceptance:**

- 重复 blind、manifest 不匹配或成本缺失均 fail-closed。
- 报告包含预测、排序、校准、传播、稳定性和成本后 Top-K 指标。
- UI/API 展示的每项正式结果均可追溯到 immutable artifact。

### Task 10: 全链路验证、远端 CI 与交付证据

**Ownership:** 验证报告、状态文件、GitHub Actions 证据；不得人为提升研究状态。

**Work:**

- 运行 Python tests/Ruff/mypy/compileall、contracts/infra、Maven、npm lint/typecheck/test/build、Compose config、release gate。
- 运行安全/secret/mock/formal-claim 扫描与合成 E2E；合成证据必须标记 non-formal。
- 仅在具备真实证据时更新 phase-state；否则保留 false 并列出可执行 blocker。
- 推送前生成 Lore 格式提交与变更清单；只有用户已授权外部推送时才推送并核验远端 SHA/CI。

**Acceptance:**

- 所有工程质量门在同一提交上全绿。
- 研究有效性状态与真实证据严格一致。
- 最终报告列出完成项、阻断项、测试证据、changed files、简化项与剩余风险。
