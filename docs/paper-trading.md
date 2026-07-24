# 股票虚拟盘自动交易

SeatFlow 现在提供一个无第三方依赖的本地虚拟券商，用于把每日预测信号接到可持久化的自动买卖闭环。它不是历史回测：每次运行都会读取当前账户、按当前报价调仓、记录成交，并把现金和持仓写回状态文件。

## 执行链路

1. 模型在交易日收盘后生成下一交易日信号。
2. 行情采集器在执行时刻生成报价快照。
3. `paper-run` 选择分数达标的 Top-N 股票。
4. 先卖出已不在目标组合中的可卖持仓，再按等权目标买入不足仓位。
5. 账户状态和每个订单批次写入 JSON；相同 `cycle_id` 重跑不会重复成交。

内置约束：

- 信号日期必须早于执行日期，避免同日未来信息。
- A 股买入数量按 100 股整数倍取整。
- 当日买入的股票当日不可卖出，按 T+1 管理持仓批次。
- 支持停牌、涨停买入阻断和跌停卖出阻断。
- 支持佣金、最低佣金、卖出印花税、滑点、现金缓冲和单票仓位上限。
- 卖出批次和买入批次分别使用幂等键，进程中断后可安全重跑。
- 完整信号、报价和配置会进入周期请求哈希；同一 `cycle_id` 不允许绑定不同输入。
- 写入锁超过 5 分钟会视为异常残留并自动恢复。

## 本地实测

```powershell
$env:PYTHONPATH = "src"

python -m seatflow paper-init `
  --paper-config configs/paper_trading/local.toml `
  --state-file artifacts/paper_trading/account.json

python -m seatflow paper-run `
  --paper-config configs/paper_trading/local.toml `
  --state-file artifacts/paper_trading/account.json `
  --signals-file configs/paper_trading/example-signals.json `
  --quotes-file configs/paper_trading/example-quotes.json `
  --cycle-id 2026-07-20-open

python -m seatflow paper-status `
  --paper-config configs/paper_trading/local.toml `
  --state-file artifacts/paper_trading/account.json `
  --quotes-file configs/paper_trading/example-quotes.json
```

重复执行相同 `paper-run` 命令时，返回结果中的 `replayed` 会变为 `true`，现金和持仓不会再次变化。

## 信号文件

```json
{
  "as_of_date": "2026-07-17",
  "generated_at": "2026-07-17T18:30:00+08:00",
  "signals": [
    {"stock_id": "600000.SH", "score": "0.72"}
  ]
}
```

生产接入时，应由预测任务直接生成该文件。`generated_at` 必须带时区。

## 行情文件

```json
{
  "trade_date": "2026-07-20",
  "captured_at": "2026-07-20T09:31:00+08:00",
  "quotes": [
    {
      "stock_id": "600000.SH",
      "price": "12.30",
      "tradable": true,
      "limit_up": false,
      "limit_down": false
    }
  ]
}
```

当前本地虚拟券商使用快照价模拟市价成交。接真实券商模拟盘时，保留 `PaperTradingService`，新增实现 `PaperBroker` 协议的适配器，把 `snapshot` 映射到账户/持仓查询，把 `submit_market_orders` 映射到券商委托与成交查询即可。QMT、同花顺、东方财富等平台的登录、订单字段和回报接口不同，需要确定目标平台后实现对应适配器，不能共用一套未经验证的 UI 自动化代码。

## 自动调度

`paper-run` 是单次、幂等执行入口，适合交给 Windows 任务计划程序或 CI 调度：

- 收盘后：生成信号文件。
- 下一交易日 09:31：生成报价文件并执行 `paper-run`。
- 盘后：执行 `paper-status` 保存账户快照。

调度器必须为每个执行日提供稳定且唯一的 `cycle_id`，例如 `2026-07-20-open`。
