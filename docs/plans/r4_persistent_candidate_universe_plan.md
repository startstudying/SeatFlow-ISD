# SeatFlow R4：持久候选池与衰减席位信号计划

## 一、R4 研究假设

当前每日候选池仅包含当日满足条件或进入龙虎榜的股票，导致：

```text
昨日 Top-5
→ 次日多数股票退出候选池
→ 无法获得今日预测分数
→ Rule A 强制卖出
→ 换手接近 100%
→ 正 gross Alpha 被成本侵蚀
```

R4 假设为：

> 将当日候选池改为包含历史事件股票和已有持仓的持久候选池，并为没有新龙虎榜事件的股票维护衰减席位信号，可以显著降低候选池退出导致的强制换手。

R4 不是继续调节 L0/L1/L2，也不是增加 GRU、FiLM、Mamba 或超图。

---

# 二、状态冻结

保留 R3.6 原始结论：

```toml
r3_6_selected_strategy = null
r3_6_task48_allowed = false
r3_6_model_valid = false
```

新增状态：

```toml
r4_hypothesis = "persistent_candidate_universe"
r4_reason = "candidate_exit_dominates_turnover"
r4_independent_validation = false
```

R3.6 的所有结果继续保持只读，不允许：

* 降低原来的 30% 换手门槛；
* 事后将 S1/P1 标为已通过；
* 修改 Rule A 后重新覆盖原结果；
* 用旧结果挑选最优持有天数或衰减参数。

S1/P1 只能作为 R4 的研究线索，因为它通过了 7 项门禁中的 6 项，并不能成为正式有效策略。

---

# 三、Task 51：计算 Rule A 的结构性换手下界

新增：

```text
src/seatflow/evaluation/turnover_floor.py
scripts/analyze_candidate_turnover_floor.py
```

每日计算：

```text
previous_holdings = H_(t-1)
current_candidates = C_t

candidate_exit_count =
|H_(t-1) - C_t|

candidate_exit_turnover_floor =
candidate_exit_count / 5
```

进一步拆分实际换手：

```text
total_turnover
=
candidate_exit_turnover
+ voluntary_rank_turnover
+ untradable_turnover
+ corporate_action_turnover
```

输出：

```text
mean_candidate_exit_turnover_floor
median_candidate_exit_turnover_floor
full_candidate_exit_day_ratio
candidate_pool_temporal_jaccard
candidate_pool_retention_rate
maximum_possible_reduction_under_rule_A
```

关键问题：

> 即使禁止所有主动换仓，在 Rule A 下理论上最低能把换手降到多少？

如果理论最大降幅本身低于 30%，则证明 R3.6 的换手门禁在 Rule A 下结构性不可达。但这不能改变 R3.6 结论，只能作为启动 R4 的依据。

---

# 四、Task 52：补齐 Point-in-Time 数据时间字段

未来生成的每条记录必须保存：

```text
trade_date
event_time
source_publish_time
source_available_time
feature_generated_time
prediction_generated_time
earliest_trade_time
```

市场执行字段至少包括：

```text
is_suspended
is_limit_up_at_entry
is_limit_down_at_exit
is_tradable_at_entry
is_tradable_at_exit
open_price
close_price
turnover_amount
market_cap
industry_id
```

必须满足：

```text
source_available_time <= prediction_generated_time
prediction_generated_time < earliest_trade_time
```

缺少精确时间字段的旧数据可用于开发分析，但不能支持严格的 future-blind 有效性声明。

---

# 五、Task 53：构造持久候选池

当前候选池：

```text
C_t = 当日龙虎榜候选股票
```

R4 候选池定义为：

```text
U_t =
当日龙虎榜股票
∪ 最近 H 日龙虎榜股票
∪ 昨日实际持仓
```

其中：

```text
H 不允许在 R3.6 数据上搜索
```

推荐通过训练期事件衰减分析确定 H：

1. 只使用训练区间；
2. 统计龙虎榜事件后第 1～5 日 gross residual return；
3. 计算各 horizon 的事件增量；
4. 选择边际增量首次低于交易成本的最短 horizon；
5. 在验证和测试期冻结。

若必须预先固定一个值，使用：

```toml
maximum_event_age_days = 5
```

因为 SeatFlow 的原研究目标本身覆盖未来 1～5 个交易日。其他 H 只能作为敏感性报告，不能用于选择。

---

# 六、Task 54：建立衰减席位状态

股票在没有新龙虎榜事件时，不再直接失去席位信息，而是保留最近一次席位状态。

为每只股票保存：

```text
last_seat_event_date
days_since_last_seat_event
last_net_buy_amount
last_seat_count
last_institution_ratio
last_seat_concentration
last_seat_signal
has_new_seat_event
```

衰减形式：

```text
decayed_seat_signal =
last_seat_signal × decay(days_since_last_event)
```

第一版使用固定指数衰减：

```text
decay(age) = exp(-age / tau)
```

`tau` 只能根据训练区间的事件收益衰减估计，不能在旧 OOS 上选择。

如果训练数据不足，固定：

```toml
seat_signal_half_life_days = 2
maximum_event_age_days = 5
```

达到最大事件年龄后，席位信号归零，但股票仍可依赖量价、行业和市场特征进行评分。

---

# 七、Task 55：将席位图改成席位聚合特征

R3.5 的推理遮蔽显示，去除 Seat–Stock 后 L2 gross spread 提升，说明当前图传播可能有害。

因此 R4 第一版不使用：

```text
Seat 节点
Seat–Stock 消息传播
Seat–Seat 共现边
```

改为股票节点上的 Point-in-Time 聚合特征：

```text
seat_buy_amount
seat_sell_amount
seat_net_amount
seat_count
institution_seat_ratio
northbound_seat_ratio
seat_concentration
seat_repeat_participation
days_since_last_event
decayed_seat_signal
```

模型结构先采用：

```text
股票量价
+ 行业
+ 市场状态
+ 衰减席位聚合特征
→ LightGBM/CatBoost Ranker
或轻量 MLP
```

先证明席位聚合特征有稳定增量，再考虑恢复图结构。

这是比继续训练当前 HGT 更可解释的路线。

---

# 八、Task 56：为池外持仓每日生成分数

当前 Rule B 无法执行，原因是退出当日候选池后没有新的股票分数。

R4 必须保证：

```text
昨日持仓 ⊆ 今日可评分股票集合
```

即使股票当天没有新的龙虎榜事件，也要通过：

```text
最新量价特征
+ 最新行业特征
+ 最新市场特征
+ 衰减席位特征
```

生成今日分数。

因此股票不再因为：

```text
没有再次进入龙虎榜
```

而被强制卖出。

真正的强制卖出只能来自：

```text
停牌或长期不可交易
退市
风险规则
数据严重缺失
超过预注册最大持有期限
```

---

# 九、Task 57：实现真正的 Rule B

组合保持 5 只股票。

每日决策：

1. 更新所有持仓股票的当前分数；
2. 更新持久候选池内新股票的分数；
3. 估计继续持有的 gross utility；
4. 估计换入新股票后的 net utility；
5. 只有净增益覆盖切换成本时才换仓。

换仓条件：

```text
expected_return_new
- expected_return_old
>
buy_cost_new
+ sell_cost_old
+ slippage
```

同时固定：

```toml
maximum_voluntary_replacements = 1
candidate_exit_forces_sell = false
```

已有持仓退出当日龙虎榜候选池，不再构成强制卖出。

---

# 十、Task 58：先做无收益结构预检

在训练或比较收益前，只检查候选池和换手结构。

输出：

```text
persistent_candidate_pool_size
candidate_pool_temporal_jaccard
candidate_exit_turnover_floor
simulated_rule_b_turnover
average_retained_holdings
full_turnover_day_ratio
```

硬门禁继续保持原 R3.6 标准：

```text
相对 P0 的换手下降 >= 30%
```

这次不降低门槛。

只有持久候选池从结构上实现至少 30% 换手下降，才允许进入经济实验。

若结构预检仍无法达到 30%，立即停止 R4，不训练模型。

---

# 十一、Task 59：建立新的未污染实验协议

R1～R3.6 的旧 folds 已用于：

* 损失选择；
* 图关系诊断；
* 成本策略选择；
* 持仓策略比较；
* 候选池问题诊断。

不能再次把这些 folds 当独立验证。

R4 分成：

```text
训练期：
估计 scaler、事件衰减和模型参数

开发期：
只比较一个持久候选池模型与一个无席位基线

future-blind：
模型冻结后第一个未查看标签的交易日起
```

最多允许两个模型：

```text
B0：股票 + 行业 + 市场
B1：B0 + 衰减席位聚合特征
```

这样直接检验席位信息是否有增量。

禁止同时比较：

* 多种 H；
* 多种 tau；
* 多种持有期；
* 多种模型；
* 多种换仓阈值。

---

# 十二、Task 60：R4 开发期门禁

B1 必须同时满足：

```text
持久候选池换手相对每日 P0 下降 >= 30%
6 次运行中至少 4 次 net spread > 0
两个 fold 的平均 net spread >= 0
平均 gross spread >= 0
平均 net spread > B0
最差运行不低于 B0
席位特征消融后性能下降
```

只有满足这些条件，才设置：

```text
r4_development_candidate = true
model_valid = false
```

如果 B1 不优于 B0，结论为：

```text
seat_information_increment = unsupported
```

SeatFlow 应转为：

```text
龙虎榜关系分析系统
席位行为画像系统
异常资金传播监控系统
```

而不是继续宣称席位数据提高选股收益。

---

# 十三、Task 61：最终 Future-Blind

冻结：

```text
数据 schema
事件年龄 H
衰减 tau
模型
五个种子或单一确定性模型
组合规则
手续费
持仓规则
```

future-blind 期间每天保存：

```text
当日持久候选池
所有候选分数
昨日持仓分数
换仓决策
预计成本
实际预测文件哈希
数据可用时间
```

只有在标签实现后一次性评估。

不得因某天没有龙虎榜事件而重新调整模型。

---

# 十四、R4 停止条件

出现任一情况即停止：

1. 持久候选池仍无法把换手降低 30%；
2. B1 不优于无席位基线 B0；
3. 席位聚合特征仅提高 RankIC、不提高净收益；
4. 收益仍依赖单一 fold 或少数日期；
5. future-blind 净收益为负；
6. 必须重新启用有害的 Seat–Stock 消息传播才能转正；
7. 必须在旧 OOS 上搜索 H、tau 或 margin 才能通过。

最终状态：

```text
model_valid = false
research_status = completed_with_negative_alpha_result
```

保留 SeatFlow 的分析、解释与可视化价值。
