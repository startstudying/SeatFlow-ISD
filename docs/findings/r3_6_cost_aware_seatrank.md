# SeatFlow R3.6 Cost-Aware SeatRank 结论

## 决策

R3.6 在冻结的开发数据上得到负结果：

- `selected_strategy = null`
- `task48_allowed = false`
- `research_status = negative_result_under_current_data`
- 不进入 Task 48、Task 49 或 Task 50
- 不写入 `model_valid`
- checkpoint 数量在回放前后均为 18

本结论绑定父提交 `0e2ccfb`、Task 25 的 144 个冻结运行文件和 18 个既有
checkpoint。R3.6 只重放冻结预测，没有重新训练模型。

## 口径与时间持仓审计

- 384 个策略运行收益摘要、共 960 个收益口径记录均同时输出 decimal、
  percentage、bps、聚合方法和天数。
- 日均收益与累计 sum/compound 被定义为不同聚合，未混用。
- 已对 12 个 L0/L2 冻结运行完成逐日 Top-5/Top-10 持仓连续性、Jaccard、
  retained/replaced、双边与单边换手、持有期和退出原因审计。
- `candidate_exit` 与 `rank_change` 可由冻结数据识别；`untradable`、
  `forced_sell`、`corporate_action`、`missing_market_data` 因字段缺失明确标为
  unavailable，未作推断。
- Candidate Exit Rule A 已执行；Rule B 因没有池外分数不可用。

## 1x 成本硬门槛结果

| 候选 | 净收益均值 | 毛收益均值 | 成本差下降 | 换手下降 | 正净运行 | 通过 |
|---|---:|---:|---:|---:|---:|---|
| S0/P1 | 0.2416% | 0.2587% | 64.61% | 13.34% | 4/6 | 否 |
| S0/P2 | 0.1461% | 0.1761% | 37.87% | 7.89% | 3/6 | 否 |
| S0/P3 | 0.1561% | 0.1865% | 37.07% | 7.73% | 3/6 | 否 |
| S1/P1 | 0.0230% | 0.0662% | 30.37% | 7.71% | 4/6 | 否 |
| S1/P2 | -0.0085% | 0.0418% | 18.95% | 4.84% | 3/6 | 否 |
| S1/P3 | -0.0089% | 0.0417% | 18.39% | 4.70% | 3/6 | 否 |

`S1/P1` 最接近门槛：正净运行、双 fold 非负、毛收益非负、净收益超过 P0、
成本下降至少 30%、最差运行不弱于 P0 均通过；但换手仅下降 7.71%，显著低于
预注册的 30%，所以仍然失败。

`S0/P1` 虽有更高的平均净收益，但 fold 1 为负、最差运行弱于 P0，且换手仅下降
13.34%，同样不能晋级。其余候选还存在正净运行不足、净收益不超过 P0或成本下降
不足等失败项。

## 成本模型

成本在人民币层拆分为佣金、印花税、买入滑点、卖出滑点和最低佣金，再除以组合
名义本金转换为小数收益率。默认 1x 参数的有效单边成本为买入 8 bps、卖出
18 bps；0x、2x、5x 仅用于压力测试，只有 1x 参与选择。每日均验证：

`net spread = gross spread - candidate cost + baseline cost`

## 停止条件

预注册规则要求只有唯一通过硬门槛的策略才允许 Task 48 的 12 次训练。本轮没有
候选通过，因此按计划停止。Task 49 独立回测与 Task 50 未来时间/可交易性 schema
均是 Task 48 之后的条件任务，本轮不得提前执行。

证据文件：

- `artifacts/r3_6/r3_5_frozen_manifest.json`
- `artifacts/r3_6/return_unit_audit.json`
- `artifacts/r3_6/temporal_holdings.json`
- `artifacts/r3_6/strategy_replay.json`
- `artifacts/r3_6/strategy_daily_results.json`
- `artifacts/r3_6/strategy_gate_decision.json`
