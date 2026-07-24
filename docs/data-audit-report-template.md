# 真实数据审计报告模板

本模板用于记录真实数据是否可进入证据晋级轨。它不替代原始快照、manifest、
交易日历或 provider 请求日志。

## 审计身份

- 审计 ID：
- 审计日期与时区：`Asia/Shanghai`
- 审计人/执行环境：
- provider 与 source version：
- 交易日历来源、版本与 hash：
- 审计区间：

## 访问与席位范围

- `daily` live endpoint：通过 / 失败
- `top_list` live endpoint：通过 / 失败
- `top_inst` 或其他席位 endpoint：通过 / 失败
- `seat_detail_scope`：`complete_seat` / `institutional_only` / `synthetic` / `none`
- 完整席位数据合同位置（仅 `complete_seat`）：
- Token：只记录“已轮换/运行时注入”，不得写入值、截图或日志。

## 覆盖与质量

- 交易日历覆盖：
- 日线/复权因子覆盖：
- 龙虎榜摘要覆盖：
- 席位明细 endpoint 覆盖：
- 空表日期及解释：
- 缺失 `available_time` 数：
- 单位口径（成交量、成交额、买卖金额）：
- 数据、manifest、配置 hash：

## 时点与可交易性

- 每个数据集的 `available_time` 假设及证据：
- 复权规则与版本：
- 停牌、涨跌停、ST、退市、新股、公司行为来源：
- 回测最早执行时点：

## 结论

- `data_ready`：true / false
- `full_seat_ready`：true / false
- 允许的 `model_validity_scope`：
- 阻塞原因：
- 后续行动：

`seatflow audit-data` 生成的 JSON 报告是可机读证据；本模板用于补充数据合同、
来源发布时间和人工审计说明。sample、mock、单日成功或 HTTP 2xx 不构成真实数据就绪。
