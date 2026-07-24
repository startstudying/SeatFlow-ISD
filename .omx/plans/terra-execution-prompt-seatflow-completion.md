# Terra Execution Prompt - SeatFlow Completion

Primary executor: `gpt-5.6-terra`

工作目录：`D:\SeatFlow`

先按顺序读取：
1. `.omx/plans/prd-seatflow-terra-completion.md`
2. `.omx/plans/test-spec-seatflow-terra-completion.md`
3. `.omx/plans/reviews/architect-seatflow-terra-completion.md`
4. `.omx/plans/reviews/critic-seatflow-terra-completion.md`
5. `.omx/plans/ralplan-handoff-seatflow-terra-completion.md`

执行目标：按已批准的工程实现轨和证据晋级轨完成 SeatFlow 剩余范围。不得把
sample/mock、单日数据、HTTP 2xx、单折/单 seed、UI 截图或本地 build 当成真实
数据、模型、回测、部署或竞赛完成证据。

硬约束：
- 保留当前 dirty/untracked 工作，不 reset、不覆盖用户改动。
- 截图中暴露的 Tushare token 已失效；真实调用前只能使用轮换后的运行时 secret，
  不得输出、记录或提交 token。
- `seat_detail_scope` 只能来自 live audit。`institutional_only` 只能发布机构范围
  结论；`synthetic/none` 只能 smoke 或阻塞；没有完整席位数据时保持 fail-closed。
- 第一次可比较训练前冻结 blind split、purge/embargo、主指标、baseline、seed、
  搜索预算、失败 run 处理和 append-only lineage。
- 特征、边、标签、预测和成交都必须绑定 `cutoff_time`、`available_time`、数据
  manifest hash；龙虎榜收盘后信号不得在同日收盘成交。
- 每阶段只在测试门槛通过后形成 Lore 格式提交；不提交 data、secret、大 checkpoint、
  MLflow 大 artifact。

执行顺序：
1. M0：质量基线、doctor、跨栈兼容矩阵、计划文件可跟踪性。
2. 并行启动 M1 live audit 与 M2 deterministic fixture 工程。
3. M2.5：首次正式 baseline 前冻结 protocol/blind/lineage。
4. 证据轨按 M1 -> M2 -> M2.5 -> M3 -> M4 -> M5 -> M6 晋级。
5. 工程轨按 M2 -> M3/M4 smoke -> M7 -> M8 -> M9 -> M10 推进。
6. M11 仅在真实盲测和提交合同齐全时执行。

每阶段必须输出：
- changed files
- commands and tests run
- evidence artifact paths and hashes
- `engineering_complete/data_ready/model_valid/backtest_valid/competition_ready`
- `model_validity_scope`
- blocked claims and remaining risks

推理/模型/回测相关阶段使用高推理强度；普通脚手架和文档阶段可用中等强度。
独立 verifier 必须在阶段状态置 true 前复核证据。

Codex App/native subagent：使用 `gpt-5.6-terra` executor 执行本文件。
OMX CLI：可由 `$ultragoal` 持久化主目标，并在 M7-M10 用 `$team` 做受控并行；
`$ralph` 仅作明确要求下的单 owner fallback。

