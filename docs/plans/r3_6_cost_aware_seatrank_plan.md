# SeatFlow R3.6：Cost-Aware SeatRank 优化计划

## 冻结结论

R3.6 以 commit `0e2ccfb` 和全部 R3.5 证据为只读输入：

```toml
task25_selection = "none"
task26_allowed = false
optimization_randomness = "unsupported"
top5_boundary_noise = "unsupported"
regime_drift = "insufficient_evidence"
turnover_and_cost = "confirmed"
seat_stock_increment = "negative_evidence"
target_misalignment = "unsupported"
```

R3.6 不增加 GRU、FiLM、Mamba、超图、HGT 层、新排序损失或随机种子。

## 强制执行顺序

1. Task 41：收益尺度与聚合审计；
2. Task 42：跨日 Top-5/Top-10 持仓与换手来源审计；
3. Task 43：冻结开发协议；
4. Task 44：实现 P0/P1/P2/P3 固定策略；
5. Task 45：在 0×/1×/2×/5× 成本下回放；
6. Task 46：按候选池退出规则 A 回放，规则 B 标记为缺少候选池外分数；
7. Task 47：只用 1× 成本应用晋级门禁；
8. 仅当一个策略通过时，Task 48 才允许产生 12 个正式训练 checkpoint；
9. Task 49：原生与独立参考回测逐日一致性；
10. Task 50：冻结未来时间与可交易性字段契约。

Task 47 前新增训练 checkpoint 必须为 0。

## Task 41：收益单位

所有收益同时输出：

```text
decimal_return
percentage_return
basis_points
aggregation_method
number_of_days
```

R3.5 的 `0.152863` 必须标明是对真实 executable residual return 的逐日
Top-5-minus-remainder spread 求算术平均，不是累计收益或标准化标签收益。

## Task 42：跨日持仓

对每个冻结运行输出 seed/temporal Jaccard、保留/替换数、单边及买卖换手、
平均持有天数和持有期分桶。卖出原因只允许：

```text
candidate_exit
rank_change
untradable
forced_sell
corporate_action
missing_market_data
```

冻结数据缺少可交易性与公司行动字段时，相关原因必须标记为不可观测。

## Task 43–46：冻结策略与成本

仅使用 S0=L0 与 S1=L2。每个信号运行 P0–P3：

- P0：每日精确 Top-5；
- P1：每日最多一次自愿替换，候选池退出属于预注册强制替换；
- P2：买入前 5、持有前 10；
- P3：Ridge walk-forward gross-return 校准，只读取当前日期前已成熟标签。

固定参数：

```toml
buy_rank = 5
hold_rank = 10
switch_margin = 0.0
cost_multiplier = 1.0
ridge_alpha = 1.0
candidate_exit_rule = "A"
```

规则 B 因缺少候选池外实时分数而只记录为后续数据需求。

成本必须拆分佣金、印花税、滑点和最低佣金，并验证：

```text
net_spread = gross_spread - candidate_total_cost + baseline_total_cost
```

只有 1× 成本参与选择；0×、2×、5× 仅用于压力报告。

## Task 47：晋级

同一信号内相对 P0，策略必须同时满足：

- 6 次运行中至少 4 次 net spread > 0；
- 两个 fold 的三种子平均 net spread 均 >= 0；
- 平均 gross spread >= 0；
- 平均 net spread > P0；
- 成本差与换手率均相对 P0 下降至少 30%；
- 最差运行不比 P0 恶化。

排序顺序固定为最差 fold、正收益运行数、种子标准差、换手率、平均 net。
如果没有策略通过，设置负研究结论并停止，禁止 Task 48。

## 验证

- 全部新增及原测试通过；
- Ruff 与严格 mypy 通过；
- Task 25/R3.5 哈希不变；
- Task 47 前无新 checkpoint；
- 产物和 commits 与 Task 25/R3.5 分离。
