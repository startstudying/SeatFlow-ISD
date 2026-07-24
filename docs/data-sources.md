# 数据源与权限策略

## 目标

SeatFlow 的训练输入需要同时具备行情、龙虎榜汇总、席位级明细以及行业/概念映射。不同数据源的覆盖能力不同，因此采集层按“能力探测、原始快照、明确降级”设计，而不是把缺失数据填成零或假席位。

## 提供方优先级

| 提供方 | 日线 | 龙虎榜汇总 | 席位明细范围 | 当前用途 |
| --- | --- | --- | --- | --- |
| Tushare Pro | 是 | `top_list` | `top_inst` 机构明细，受权限限制 | 首选结构化源 |
| AKShare | 是 | 是 | 当前未验证完整适配 | 免费降级与校验 |
| SampleProvider | 合成 | 合成 | 合成 | 离线测试，不得用于研究结论 |

Tushare REST 的官方契约是向 `http://api.tushare.pro` POST JSON，字段为 `api_name`、`token`、`params`、`fields`。本项目只在运行时从 `TUSHARE_TOKEN` 读取 Token，任何错误消息、日志、manifest 和示例文件都不得输出 Token。由于官方示例 URL 使用 HTTP，采集前必须额外设置 `TUSHARE_ALLOW_INSECURE_HTTP=1`；否则项目会拒绝发送 Token。

官方资料：

- [Tushare HTTP 调用契约](https://tushare.pro/document/1?doc_id=130)
- [Tushare 龙虎榜每日明细 top_list](https://tushare.pro/document/2?doc_id=106)
- [Tushare 龙虎榜机构明细 top_inst](https://tushare.pro/document/2?doc_id=107)
- [AKShare 股票数据接口](https://akshare.akfamily.xyz/data/stock/stock.html)

## 权限与降级

1. `top_list` 只提供股票级龙虎榜汇总，不能单独构造 Seat -> Stock 边。
2. `top_inst` 只覆盖机构席位明细，且官方权限说明存在页面差异；实际请求返回权限错误时，CLI 会保留已成功的日线和汇总快照，并报告席位数据不可用。
3. 当没有有效席位级数据时，`build-artifacts` 不会伪造图快照。特征产物仍可生成，但完整 DT-SeatHGT 训练必须停止在数据门槛前。
4. AKShare 公开网页数据仅用于研究和校验。接口字段、历史范围、网站可用性都要在每次采集后以 manifest 和质量报告确认。

## 快照布局

```text
data/
├─ raw/<provider>/<dataset>/date=YYYY-MM-DD/data.json
├─ raw/<provider>/<dataset>/date=YYYY-MM-DD/manifest.json
├─ processed/<provider>/<dataset>/date=YYYY-MM-DD/data.json
├─ features/<provider>/stock_feature/date=YYYY-MM-DD/data.json
└─ graphs/<provider>/hetero_graph/date=YYYY-MM-DD/data.json
```

`manifest.json` 包含提供方、数据集、交易日、行数、SHA-256、写入时间与质量报告。原始数据目录被 Git 忽略，避免数据、凭据或大文件进入仓库。

## 时间语义

- `trade_date`：记录归属交易日。
- `available_time`：在研究/交易流程中最早可用的时间。
- `written_at`：本机写入快照的时间，不可用于替代 `available_time`。

默认的 `data.available_after_close = "18:00"` 是保守的配置占位，不是对任一数据源发布时间的事实断言。接入真实源前必须按数据供应商公告、采集日志和交易规则确认，并调整为实际可用时间。

日线的成交量和成交额保留 `volume_unit`、`amount_unit`。当前 Tushare 保存其接口原始单位 `hand` 与 `thousand_cny`；AKShare 保存为 `provider_reported`，在未核验单位前不得和其他来源混合计算流动性特征。
## 真实数据审计

真实数据进入训练、回测或对外展示前，必须用经验证的交易日历执行：

```powershell
$env:PYTHONPATH = "src"
python -m seatflow audit-data --provider tushare --start 2021-01-01 --end 2026-01-01 --trading-calendar-file data/calendar/trade_dates.json --require-seat-detail
```

命令会生成 `artifacts/data_audit/*.json`。没有轮换后的 Token、verified trading
calendar、完整可用时间证据或 seat-detail 权限时，报告保持 `data_readiness=false`。
`top_inst` 只允许 `institutional_only` 范围结论；只有另行授权并通过 live audit 的
完整券商席位数据才可标记为 `complete_seat`。
