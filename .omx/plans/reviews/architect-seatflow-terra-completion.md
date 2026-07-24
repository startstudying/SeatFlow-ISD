# Architect Review - SeatFlow Terra Completion

审查顺序：Planner draft -> Architect round 1 -> plan revision -> Architect round 2。

## Round 1

Verdict: ITERATE

主要阻塞：
- `seat_detail_scope` 未绑定允许声明，存在把 `institutional_only` 误报为完整席位的风险。
- blind split、主指标、baseline、seed 和搜索预算冻结过晚。
- 最小 lineage 在正式 baseline/DT 训练之后才建立。
- 缺 PyG/MLflow/Java/Next/Docker 跨栈兼容矩阵和 live data 量化阈值。
- `.omx/` 被整体忽略，计划不具备可版本化交接能力。

## Revision Evidence

- PRD 新增 `seat_detail_scope` 晋级矩阵、工程实现轨/证据晋级轨和数据/服务边界 ADR。
- M2.5 在第一次可比较训练前冻结 blind/protocol/lineage。
- M0 新增 doctor、依赖矩阵、PyG CPU smoke 和 MLflow tracking extra 约束。
- M1 新增数据覆盖、历史长度、样本量和少数类比例阈值。
- M6 新增下一交易日执行、T+1、整数手、停牌/涨跌停、ST/退市/新股、幸存者偏差和成本敏感性约束。
- `.gitignore` 精确放行指定计划与 review 文件，继续忽略其他 `.omx` 运行状态。

## Round 2

Verdict: APPROVE

Architect review: APPROVE。第二轮修订已解决上一轮阻塞项：计划新增了
`seat_detail_scope` 晋级矩阵，明确 `institutional_only` 只能支持机构范围结论，
`synthetic/none` 只能用于 smoke 或阻塞声明；工程实现轨与证据晋级轨已拆分，
避免 M1 数据权限阻塞时误停工程，也避免 fixture 工程被误报为真实模型完成；
M2.5 已在首次 baseline 前冻结 blind split、purge/embargo、主指标、baseline、
seed、搜索预算和 append-only lineage；M0 已要求跨栈 doctor/版本矩阵和 Git
精确放行计划文件；M1 已补充量化数据阈值；M6 已补充 A 股真实执行约束。
当前无阻塞级架构缺口，批准作为 Terra 执行共识计划进入实现与验证阶段。

## Approved Artifacts

- `.omx/plans/prd-seatflow-terra-completion.md`
- `.omx/plans/test-spec-seatflow-terra-completion.md`
- `.omx/plans/ralplan-handoff-seatflow-terra-completion.md`

