# 数据契约（初始化版）

## 龙虎榜交易记录

最小字段：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `trade_date` | date | 交易日 |
| `seat_id` | string | 非空、稳定席位标识 |
| `stock_id` | string | 规范为 `600519.SH`、`000001.SZ` 或 `920082.BJ` |
| `side` | enum | `buy` 或 `sell` |
| `amount` | float | 大于 0，单位需全局统一 |
| `industry` | string/null | 可缺失 |
| `concepts` | list[string] | 可为空 |
| `event_reason` | string/null | 可缺失 |
| `available_time` | datetime/null | 构图时必须早于或等于预测截止时间 |
| `source_name` | string | 不可缺失，记录数据来源 |
| `source_version` | string/null | 数据源或适配器版本 |

## 防泄漏约束

- 构建日期为 `as_of_date` 的图快照时，只允许使用 `trade_date <= as_of_date` 的记录。
- 标签区间必须严格晚于特征截止日。
- `written_at` 和本机下载时间不能替代 `available_time`。
- 席位历史胜率、历史收益等聚合特征必须只使用当时可见数据。
- 数据切分必须按时间进行，禁止随机打散跨越未来区间。

## 待真实数据确认

- 龙虎榜买卖金额口径、净额口径和重复席位合并规则。
- 股票复权方式、停牌/涨跌停处理、交易日历来源。
- 行业和概念的版本日期，避免使用事后分类。
- 上榜原因编码、席位别名归一和机构席位识别规则。
- 分类标签阈值、收益标签定义和主要评估指标。
