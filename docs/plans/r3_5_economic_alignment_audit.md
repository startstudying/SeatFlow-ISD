# SeatFlow R3.5：经济目标对齐与 Top-5 尾部失效审计计划

## 目标与边界

R3.5 只分析 commit `1ac4097` 对应的 Task 25 冻结证据：

- 18 次运行（L0/L1/L2 × 2 folds × 3 seeds）；
- 144 个运行文件及其 SHA-256；
- 固定数据集、fold、随机种子、手续费与 Top-5 规则；
- 已保存预测、标签和 checkpoint。

禁止训练新模型、调整损失、增加特征、重新选择种子、修改门禁或覆盖 Task 25
文件。Task 26、GRU、FiLM 和超图继续保持禁止。

## D0：冻结 Task 25

生成 `artifacts/r3_task25/frozen_manifest.json` 与
`docs/findings/r3_task25_frozen_status.md`。冻结清单必须逐文件验证
`run_complete.json` 中记录的大小和 SHA-256，并保留以下状态：

```toml
task25_complete = true
task25_selection = "none"
task25_research_only = true
task26_allowed = false
```

## D1：Oracle 一致性

对每个交易日计算：

1. 训练标签 Oracle：按真实 residual z-score/rank 排序；
2. 可执行收益 Oracle：按真实可执行未来残差收益排序；
3. 分数方向：比较原始 `score` 与 `-score`。

两类 Oracle 均使用与 Task 25 相同的候选池、Top-5、等权组合和交易成本。
输出 `oracle_alignment.json` 与行级 `oracle_daily_results.parquet`。

## D2：收益、换手和成本归因

分别记录候选与基准的 gross return、买卖换手、成本和 net return，并验证：

```text
net_spread = gross_spread - candidate_cost + baseline_cost
```

按运行和目标汇总 gross spread、cost difference、net spread、成本解释比例、
换手率与持仓延续率。

## D3：Top-K 尾部诊断

计算 Top-1/3/5/10/20/10%/20% 的 gross/net spread、十分位组合收益、
NDCG@5/10、Precision@5、Top-5 实际收益分位、Rank 1–5 与 Rank 6–10
真实收益。所有排序以 `stock_id` 作为确定性并列决胜键。

## D4：时间对齐

审计数据交易日、可用时间、预测生成时间、入场/出场日期与价格类型、标签
horizon。若冻结数据未保存真实时间戳，必须标记为证据不足，不得推断。

固定对齐矩阵：

| 训练标签 | 回测执行 | 一致 |
| --- | --- | --- |
| T+1 close-to-close | T+1 open-to-close | 否 |
| T+1 open-to-close | T+1 open-to-close | 是 |
| T+1 close-to-T+2 close | T+1 close-to-T+2 close | 是 |

## D5：风险暴露

比较每日 Top-5 与候选池。仅使用冻结图中真实存在的字段；当前可直接审计的
核心字段包括波动率、量比、前期收益、振幅、股票出现频率和席位边代理。
市值、真实成交额/换手率、Beta、行业、涨停可买性和机构席位比例若缺失，
必须在输出中逐项列为不可审计。

## D6：关系遮蔽

在图副本上分别运行完整图、关闭 Seat–Stock、关闭 Industry–Stock、
关闭 Stock–Stock 和仅股票节点特征。比较分数变化、Top-5 Jaccard、
RankIC 和 Top-5 spread。遮蔽不得修改冻结图，也不能替代正式重训练消融。

## D7：A–F 决策

`diagnostic_decision.json` 中 A–F 每项只能使用：

- `confirmed`
- `supported`
- `unsupported`
- `insufficient_evidence`

最终报告必须把证据、限制和下一路线分开陈述。F 优先检查标签/交易区间，
D 仅在 gross 为正而成本使 net 转负时确认，E 仅在关系遮蔽证据充分时确认。

## 验证与停止条件

- 全部原测试与新增测试通过；
- Ruff 与 mypy 通过；
- Task 25 冻结哈希在审计前后相同；
- 没有新增训练 checkpoint；
- 所有要求的 JSON、Parquet 与 findings 文档存在；
- 以独立、范围明确的 commits 记录冻结、实现和结果。
