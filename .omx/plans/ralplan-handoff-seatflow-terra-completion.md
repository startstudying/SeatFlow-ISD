# RALPLAN Handoff - SeatFlow Terra Completion

计划文件：
- PRD：`.omx/plans/prd-seatflow-terra-completion.md`
- 测试规范：`.omx/plans/test-spec-seatflow-terra-completion.md`

## RALPLAN-DR

### Principles

1. 证据优先：没有真实数据、真实 checkpoint、真实回测或盲测证据时，不得声明对应完成。
2. Fail-closed：缺权限、缺字段、缺 `available_time`、缺 lineage、缺 checkpoint 时停止晋级，而不是 mock/填零/静默降级。
3. 时点一致：所有特征、边、标签、预测和回测必须绑定 cutoff 和 `available_time`。
4. 分层交付：Python 数据/模型、FastAPI、Java、Next、Docker 分阶段验收，不能用 UI 或 2xx 证明模型有效。
5. 可复现：每个实验、回测、发布包都绑定 config hash、dataset hash、code status 和 artifact hash。

### Top Drivers

1. 数据真实性与席位明细权限决定是否能进入完整 DT-SeatHGT，而不是代码完整度。
2. 训练/评估/回测必须先有强 baseline 和 lineage，否则模型表现不可解释。
3. 全栈展示必须滞后于模型/回测证据，先支持降级/空状态，再展示真实预测。

### Viable Options

#### Option A - 证据门槛双轨交付（推荐）

做法：工程实现轨允许在 M1 数据阻塞时继续用 deterministic fixture 完成可逆代码、契约、空态和降级态；证据晋级轨严格按 M1 -> M2 -> M2.5 -> M3 -> M4 -> M5 -> M6 推进五个状态位。Java/Next/Docker 只在契约稳定后并行准备，真实预测验收必须等待证据轨。

优点：
- 最小化 mock 冒充真实完成的风险。
- 与现有文档的 fail-closed 和防泄漏边界一致。
- 模型、回测、竞赛结论可审计。
- 数据权限阻塞时仍能推进工程，不会把“fail-closed”误解为“停止所有实现”。

缺点：
- 如果 token/seat detail 权限缺失，会较早阻塞模型和 UI 真实展示。
- 全栈可见产出较晚。

适用：本项目当前缺真实训练/回测/盲测，且用户明确要求不可用 Mock/样例/单折替代。

#### Option B - 工程骨架并行、数据门槛异步

做法：M0 后并行推进 Java/Next/Docker 骨架，同时另一路做 M1-M6；前端先展示降级/空态。

优点：
- 更早获得全栈联调框架和 UI 截图。
- 可提前发现 Java/Next/Docker 契约问题。

缺点：
- 容易被误判为产品完成。
- 若 UI 中出现 mock 数值，会直接违反完成口径。
- 数据/模型契约变动会增加返工。

适用：演示期限强、但必须明确标记 engineering scaffold，不得标记 model/backtest complete。

#### Option C - 先复用 reference-init 全栈草图

做法：复制 `.omx/reference-init` 中的 FastAPI/Java/Next/Docker 草图作为起点。

优点：
- 目录结构最快成形。

缺点：
- 参考包存在 `/v1/mock/prediction` 和模型 placeholder；直接合并会把 mock 风险带入主线。
- 与当前 Python 包已有边界和测试不完全一致。

结论：只允许作为 UI/目录/部署草图参考，不作为实现来源。若使用，必须删除 mock 端点和 placeholder 声明，并重新写契约测试。

### Recommended Option

推荐 Option A 作为主路线，并允许 Option B 的“骨架准备”在 M7 OpenAPI 草案之后受控并行。禁止 Option C 直接合并。任何 scope 只能由 live audit 判定：`complete_seat` 可支持完整席位声明，`institutional_only` 只能支持机构范围结论，`synthetic/none` 只能 smoke/阻塞。

## 分阶段执行摘要

详见 PRD 文件。执行时必须按以下晋级门槛判断：

1. M0 工程基线：unittest/ruff/mypy/preflight 通过。
2. M1 真实数据审计：live provider 审计并锁定 `seat_detail_scope`；sample/synthetic 不能晋级。
3. M2 标签/张量化：无泄漏、PyG HeteroData、dataset manifest。
4. M2.5 协议冻结：blind split、purge/embargo、主指标、baseline、seed、搜索预算、最小 lineage。
5. M3 基线：多 seed、多 rolling windows、强 baseline、全部失败 run 留痕。
6. M4 DT-SeatHGT：完整模型、多任务头、checkpoint、真实指标、解释稳定性。
7. M5 MLflow：导入并核验自首次训练起的 append-only lineage。
8. M6 回测/盲测：真实 OOS、下一交易日执行、A 股交易约束、冻结盲测。
9. M7 FastAPI：真实推理服务，无 checkpoint 时 fail-closed。
10. M8 Java：业务编排，不生成预测。
11. M9 Next：四页 UI，真实/降级/空态清晰。
12. M10 Docker：端到端部署 smoke。
13. M11 竞赛/研究包：提交格式、盲测、复现齐全。

## Terra 主执行与并行/顺序建议

### Terra 主执行

建议 Terra 作为单一主执行 owner，负责：
- 维护阶段状态：`engineering_complete`、`data_ready`、`model_valid`、`backtest_valid`、`competition_ready`。
- 每阶段结束写入证据清单。
- 不跨过证据晋级条件；若 M1 数据权限失败，工程实现轨仍可继续，证据状态保持 false，并转入“工程骨架/降级展示”支线。
- 维护 `artifacts/status/phase-state.json`，每个 true 状态必须绑定 evidence hash、scope 和验证时间。

### 推荐执行节奏

顺序主线：
1. M0
2. 并行启动 M1 live audit 与 M2 fixture 工程
3. M2.5 在第一次正式 baseline 前冻结
4. 证据轨按 M1 -> M2 -> M2.5 -> M3 -> M4 -> M5 -> M6
5. 工程轨按 M2 -> M3/M4 smoke -> M7-M10
6. M11 仅在证据轨通过后执行

受控并行：
- M8 Java 契约骨架可在 M7 OpenAPI 草案后启动。
- M9 Next 四页空/错误/降级态可在 M8 API schema 草案后启动。
- M10 Docker base compose 可在 M7/M8/M9 服务端口和环境变量稳定后启动。

禁止并行提前验收：
- 不得在 M1 未通过时验收 M4 模型有效。
- 不得在 M6 未通过时验收回测有效或竞赛就绪。
- 不得在 M7 无真实 checkpoint 时验收真实预测 API。

### 可用 agent 类型 roster 与 staffing 建议

若后续使用 Team + Ultragoal：
- `executor`（medium）：M0-M2 Python 实现、CLI、配置。
- `test-engineer`（medium）：测试规范落地、泄漏测试、回测 fixture。
- `debugger`（high）：PyG/CUDA/训练失败排障。
- `dependency-expert`（high）：torch/PyG/MLflow/Java/Next/Docker 版本兼容。
- `architect`（high）：FastAPI-Java-Next-Docker 边界和数据/模型声明边界审查。
- `verifier`（high）：阶段证据、完成口径、盲测/回测 claim 审查。
- `code-reviewer`（high）：M4-M10 关键 diff 审查。

建议并发：
- M0-M2：Terra 主执行 + test-engineer/verifier 小并发。
- M3-M6：Terra 主执行 + debugger + test-engineer + verifier。
- M7-M10：Terra 主执行 + executor(Java) + executor(Next) + dependency-expert(Docker/MLflow) + verifier。

## Team / Ultragoal launch hints

推荐默认：Team + Ultragoal。

示例提示：
```text
$ultragoal "按 .omx/plans/ralplan-handoff-seatflow-terra-completion.md 执行 SeatFlow 剩余交付。严格区分 engineering_complete/data_ready/model_valid/backtest_valid/competition_ready；每阶段写证据，不得用 sample/mock/单折替代。"
```

若使用 Team：
```text
$team "按 .omx/plans/prd-seatflow-terra-completion.md 和 .omx/plans/test-spec-seatflow-terra-completion.md 执行。Terra 为主 owner；先完成 M0-M1，M1 未通过不得进入真实模型有效声明。并行只允许 Java/Next/Docker 骨架准备，不得提前验收真实预测。"
```

Team verification path：
1. 每个 lane 返回 changed files、commands、evidence、blocked claims。
2. verifier 汇总五个状态位。
3. Ultragoal ledger 只 checkpoint 已有证据的状态。

Ralph fallback：
- 仅当用户明确希望单 owner 持续修复/验证时使用 `$ralph`。
- Ralph 不应替代 Team 的多栈并行能力，也不能跳过 M1/M6 门槛。

## ADR

### Decision

采用证据门槛双轨交付（Option A）作为 Terra 主路线：工程实现可以在真实数据阻塞时继续，但五个完成状态只能沿证据轨晋级。允许 Java/Next/Docker 在契约稳定后做受控骨架并行；禁止把 reference-init placeholder 当作完成代码合并。

### Drivers

1. 当前仓库明确缺 PyG/HGT 训练、真实数据、回测、盲测和全栈生产实现。
2. 数据源权限和 `available_time` 是模型有效性的硬前置。
3. 用户要求不可用 Mock/样例/单折结果替代。
4. 单纯线性阻塞会在缺 token 时停止全部工程实现，双轨能同时保证进度与声明诚实。

### Alternatives considered

- Option B：工程骨架并行。保留为受控支线，但不能提前声明真实完成。
- Option C：直接复用 reference-init。因 mock/placeholder 风险拒绝作为主路线。

### Why chosen

严格门槛路线最符合现有文档边界：`docs/data-sources.md:28` 已规定无席位级数据时不伪造图快照，`docs/training-preflight.md:17` 已声明当前无 PyG/DT-SeatHGT 训练器，`README.md:16` 已声明不得宣称训练或真实收益结论。

### Consequences

- 若真实 token/seat detail 权限缺失，项目可继续做工程骨架和降级体验，但 `data_ready=false`、`model_valid=false`、`backtest_valid=false`。
- 若模型不优于 baseline，仍可声明工程完成，但不能声明模型有效或竞赛就绪。
- 全栈展示必须展示 data_scope、lineage、risk_flags。

### Follow-ups

1. M0 修复 mypy tests 类型问题或明确 mypy 范围。
2. M1 获取轮换后的 Tushare token 并审计 endpoint 权限。
3. M0 锁定 Python/PyG/CUDA/MLflow/JDK/Node/Docker 兼容矩阵，并保持指定 `.omx/plans` 共识文件可被 Git 跟踪。
4. M2 冻结标签、复权、交易日历、停牌/涨跌停规则。
5. M2.5 在首次比较训练前冻结 blind split、主指标、baseline、seed、搜索预算和 lineage schema。
6. M11 输出 model/data/backtest/blind cards。

## Terra handoff prompt

```text
你是 Terra 主执行 agent，工作目录 D:\SeatFlow。严格按以下计划执行，不要跳过阶段门槛：

计划文件：
- .omx/plans/prd-seatflow-terra-completion.md
- .omx/plans/test-spec-seatflow-terra-completion.md
- .omx/plans/ralplan-handoff-seatflow-terra-completion.md

硬规则：
1. 不得重置或覆盖用户未提交改动。
2. 不得提交、打印或写入任何 token/secret；已有泄露 token 视为失效，必须轮换。
3. sample/mock/AKShare 单日成功/HTTP 2xx/本地 build/单折结果不能证明真实数据、模型、回测或竞赛就绪。
4. 严格维护五个状态：engineering_complete、data_ready、model_valid、backtest_valid、competition_ready。
5. 每个状态只能在对应证据存在时置 true；否则写 blocked reason。
6. 所有特征、边、标签、预测、回测必须绑定 cutoff/available_time/manifest hash。
7. .omx/reference-init 只能做参考；其中 mock endpoint 和 placeholder model 不能作为完成实现。
8. seat_detail_scope 必须来自 live audit：complete_seat 才能声明完整席位；institutional_only 只能声明机构范围；synthetic/none 只能 smoke/阻塞。
9. 第一次可比较训练前必须冻结 blind split、purge/embargo、主指标、baseline、seed、搜索预算和最小 lineage。
10. 每个里程碑使用 Lore 格式提交；未通过门槛不提交完成声明，不改写历史 evidence。

起步：
1. 先做 M0：修复/稳定质量门槛，运行 unittest、ruff、mypy、preflight，锁定跨栈版本矩阵，并确认最终计划文件可被 Git 跟踪。
2. 并行启动 M1 live 数据审计和 M2 deterministic fixture 工程。没有轮换 token 或 seat detail 权限时，只停止证据晋级，不停止可逆工程实现。
3. 在第一次正式 baseline 前完成 M2.5：冻结 blind/protocol/lineage。
4. 证据轨通过后按 M3-M6 完成真实基线、DT-SeatHGT、MLflow 核验、OOS 回测与独立盲测。
5. M7-M10 做 FastAPI、Java、Next、Docker，但任何预测 UI/API 必须有 lineage 和 risk_flags，缺模型时 fail-closed。
6. M11 只在盲测和提交合同满足后输出 competition_ready=true。

每阶段交付：
- changed files
- commands run
- evidence artifact paths
- status flags
- scope and evidence hashes
- blocked claims
- remaining risks
```

## 当前已知阻塞

- 缺轮换后的 Tushare token 和 endpoint 权限证据。
- 缺完整 brokerage-seat 历史覆盖证明；`top_inst` 仅能先按 institutional scope 处理。
- 缺 PyTorch/PyG/CUDA 当前机器兼容性验证。
- 缺比赛盲测/提交格式/截止时间。
- `mypy src tests` 当前有 3 个 tests 类型错误，M0 需处理。

## RALPLAN Consensus Record

```yaml
planning_artifacts:
  prd: .omx/plans/prd-seatflow-terra-completion.md
  test_spec: .omx/plans/test-spec-seatflow-terra-completion.md
  terra_prompt: .omx/plans/terra-execution-prompt-seatflow-completion.md
  architect_review: .omx/plans/reviews/architect-seatflow-terra-completion.md
  critic_review: .omx/plans/reviews/critic-seatflow-terra-completion.md

ralplan_architect_review:
  verdict: APPROVE
  rounds: 2
  final_finding: no_blocking_architecture_gap

ralplan_critic_review:
  verdict: APPROVE
  sequence_after_architect: true
  final_finding: implementation_ready_with_external_fail_closed_gates

ralplan_consensus_gate:
  complete: true
  review_order:
    - planner
    - architect
    - critic
  approved_for_executor: gpt-5.6-terra
```
