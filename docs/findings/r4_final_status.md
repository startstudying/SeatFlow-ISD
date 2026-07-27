# SeatFlow R4 最终状态

## 结论

R4 已实现 Task 51–58 所需的诊断与基础设施，但 Task 58 因逐日结构证据不足而
fail closed。Task 59–61 未获授权，未训练 B0/B1，未创建 checkpoint，也未作
future-blind 有效性声明。

```text
R4 stopped before economic training.
No model-selection claim was made.
No future-blind validity claim was made.
model_valid remains false.
```

## 任务状态

| Task | 状态 | 说明 |
|---|---|---|
| 51 | complete | 计算冻结 R3.6 Rule A 候选退出换手下界 |
| 52 | complete | 增加 PIT 时间与市场执行 schema，缺失时间戳时 fail closed |
| 53 | complete | 固定 H=5 的交易日持久候选池 |
| 54 | complete | 固定半衰期 2 日、最大年龄 5 日的前向衰减席位状态 |
| 55 | complete | 股票级 PIT 席位聚合特征与 train-only scaler |
| 56 | complete | 昨日持仓并入今日可评分集合，无法评分必须给出原因 |
| 57 | complete | 固定 Top-5、每日最多一次主动替换的 Rule B |
| 58 | complete / gate failed | `insufficient_evidence`，未达到可判定门禁的证据要求 |
| 59 | not allowed | Task 58 未通过 |
| 60 | not allowed | 未训练、未比较 B0/B1 |
| 61 | not allowed | 未建立或宣称 future-blind 候选 |

## 结构指标

```text
Rule A 理论平均换手下界 = 0.669570707070707
P0 平均实际换手         = 0.92739898989899
Rule A 理论最大降幅     = 0.2780122532334923（27.80%）
Rule B 模拟换手         = null
R4 相对 P0 换手下降     = null
```

Rule A 即使消除全部下界以上换手，最大理论降幅仍低于 30%。这不修改冻结的
R3.6 结论，只解释了启动 R4 的结构原因。

Task 58 不能用旧数据生成真实 Rule B 模拟换手，因为冻结数据没有持仓离开当日
事件池后的不可变逐日新分数；同时 `untradable`、`corporate_action` 等执行字段
不可用。根据预注册规则，不以旧分数、汇总值或零值填补缺口。

## 门禁状态

```json
{
  "task58_gate_passed": false,
  "evidence_complete": false,
  "task59_allowed": false,
  "task60_allowed": false,
  "task61_allowed": false,
  "r4_development_candidate": false,
  "model_valid": false
}
```

## 新增 artifacts

- `artifacts/r4/task51_turnover_floor.json`
- `artifacts/r4/task51_daily_turnover_floor.parquet`
- `artifacts/r4/task58_structure_preflight.json`
- `artifacts/r4/task58_daily_structure.parquet`
- `artifacts/r4/task58_gate_decision.json`
- `artifacts/r4/r4_status.json`

Task 58 的 parquet 是具有固定 schema 的有效空 parquet，不是伪装成 parquet 的
文本标记；空行表示没有足以支持结构回放的完整逐日证据。

## 提交

- `a40a2d2`：Task 51 换手下界代码、测试与 artifacts
- `93a0b93`：初始化 R4 状态 artifact
- `902a151`：Task 52 PIT schema
- `7646924`：Task 53 持久候选池
- `d749f87`：Task 54 衰减席位状态
- `dd9af5e`：Task 55 席位聚合特征
- `3e64c67`：Task 56 池外持仓评分
- `ca8f5b5`：Task 57 Rule B
- `731a259`：Task 58 fail-closed 结构门禁

## 验证

```text
pytest: 391 passed, 3 third-party deprecation warnings
ruff check .: passed
mypy --strict src/seatflow: passed (117 source files)
checkpoint count before Task 51: 1
checkpoint count after Task 58: 1
R3.6 frozen artifact hashes: unchanged
```

## 尚存风险

1. 旧数据缺少精确的 `source_available_time`，只能用于历史开发，不能支持严格
   future-blind 声明。
2. `untradable`、停复牌、涨跌停、公司行动和严重市场数据缺失的逐日证据不完整。
3. 当前没有持仓离开当日事件池后的冻结逐日新分数，因此不能报告真实 Rule B
   模拟换手或相对 P0 降幅。
4. 只有未来生成的 PIT 数据与每日评分 artifact 完整后，才能重新执行 Task 58；
   不得用旧 OOS 搜索 H、half-life、margin、持有期或模型家族。
