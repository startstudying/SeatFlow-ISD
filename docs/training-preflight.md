# 训练前置门槛

`seatflow preflight` 会校验 TOML 中训练、验证、测试区间严格按时间先后排列。它不替代真实数据审计。

## 必须通过的门槛

1. 训练结束日早于验证开始日，验证结束日早于测试开始日。
2. 图构建传入 `cutoff_time` 时，只有 `available_time <= cutoff_time` 的边可进入图；缺失可用时间的记录会被拒绝。
3. 每个数据集均有 manifest、质量报告和来源字段。
4. 使用 `top_list` 而没有席位明细时，不能将产物描述为完整 SeatHGT 图。
5. 标签、复权口径、交易成本、停牌和涨跌停规则仍需在真实训练前配置并记录。

用于训练的采集任务应启用 `seatflow ingest --require-seat-detail`。若 `top_inst` 权限、网络或数据覆盖不满足，该命令会失败而不是返回可误判为完整数据日的成功状态。

## 当前未实现的训练声明边界

- 尚未安装或实现 PyTorch Geometric 的 HGT/DT-SeatHGT 训练器。
- 尚未有真实 Tushare/免费源下载的历史数据、盲测结果或回测收益。
- 当前 `HeuristicSmokePredictor` 仅验证接口结构和图数据流，不能用于预测表现或投资决策。
