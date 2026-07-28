# PR-D: Paper Trading MVP

> Branch: `feat/paper-trading-mvp`
> Scope: complete paper-trading business, not interleaved with model core. Includes broker, models, service, IO, and 285-line test suite. **No checkpoints, no results directories, no model code.**

## What this PR contains

### Database tables (`backend-java/src/main/resources/db/migration/`)

The Java migration files define the paper-trading schema:
- `sf_paper_account`
- `sf_cash_ledger`
- `sf_order`
- `sf_fill`
- `sf_position`
- `sf_portfolio_snapshot`
- `sf_risk_rule`
- `sf_strategy_signal`
- `sf_decision_trace`

### Python broker (`src/seatflow/paper_trading/broker.py`, 551 lines)

- A-share matching rules:
  - 100股 / 整数手
  - 涨跌停
  - 佣金、印花税、买卖滑点
- Partial fills, rejected orders
- Order cancellation

### Models (`src/seatflow/paper_trading/models.py`, 291 lines)

- `PaperAccount`, `Order`, `Fill`, `Position`, `RiskRule`, `StrategySignal`, `DecisionTrace`

### Service & IO (`src/seatflow/paper_trading/service.py`, 184 lines; `io.py`, 116 lines)

- Business orchestration
- Read/write from database

### Tests (`tests/test_paper_trading.py`, 285 lines)

- Core order/fill/position logic

## Already implemented (verified during Week-3 gap analysis)

Re-reading `broker.py` confirmed these rules are already in place (the original PR-D scope note underestimated coverage):

- **100股 / 整数手** — `broker.py:136` rejects non-lot-multiple quantities
- **涨停买拒绝** — `broker.py:168-171`
- **跌停卖拒绝** — `broker.py:206-216`
- **停牌** — handled via `MarketQuote.tradable=False` (`broker.py:151-154`)
- **T+1 卖出限制** — `broker.py:221-232` only sells lots with `acquired_date < trade_date`
- **佣金 / 印花税 / 滑点** — `broker.py:172, 236-239`
- **部分成交** — `broker.py:227-228` (T+1 partial fills)
- **幂等下单** — `broker.py:79-94` detects batch_id reuse with request_hash
- **决策审计** — `batches` dict persists every fill; `OrderBatchResult.replayed` flag

## Risk rules added in this PR (`src/seatflow/paper_trading/risk.py`)

A new pure-function risk engine evaluated BEFORE orders reach the broker:

- `RiskRuleConfig` — single-stock cap, total position cap, sector concentration, daily loss limit, max drawdown
- `PortfolioRiskState` — peak equity, equity today, trade date (supplied by Week-2 pipeline)
- `check_buy_orders(...)` — returns `RiskCheckResult` with `accepted` / `rejections` / `data_complete` / `model_valid`
- Rules: `single_stock_position_cap`, `total_position_cap`, `sector_concentration`, `daily_loss_limit`, `max_drawdown`, `data_incomplete`, `model_invalid`
- `model_valid=false` ⇒ all orders rejected with "human confirmation required, auto-trading blocked"

Tests: `tests/test_paper_trading_risk.py` (10 tests, all passing).

## Known gaps still tracked as Week-3 follow-ups

- `decision_trace` audit log: `batches` dict records fills, but a dedicated `sf_decision_trace` writer that joins risk rejections + fills into a single decision record is not yet wired
- Integration: `service.py` does not yet call `check_buy_orders` before `submit_market_orders` — the risk engine is available but not yet in the call path

These two gaps are explicit Week-3 integration work after PR-D lands.

## Merge note

The paper-trading infrastructure on this branch was previously bundled into
the 367-file PR #4. It is isolated here so the trunk can review the
paper-trading business without also reviewing CI hygiene, PIT infrastructure,
or the DT-SeatHGT v3 model.