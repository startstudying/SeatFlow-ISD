# SeatFlow-ISD V1.0 构建设计

## 目标

以《SeatFlow_ISD_数据集构建设计_V1.0》为冻结业务契约，把现有真实 Tushare 快照接入 `research-engine` 的 Dataset v3 模块，生成可追溯、可复现、默认 fail-closed 的 SeatFlow-ISD V1.0 数据集目录。

本次交付先完成“数据集本身”，不把模型有效性、回测有效性或竞赛就绪状态混入数据集完成判定。

## 输入边界

- 主数据根：`data/historical_backfill/tushare_2023_2026_v2`。
- 已确认基线窗口：2023-01-03 至 2026-07-17，共 856 个交易日快照；正式构建前须以同一真实来源增量回补到 2026-08-06（含）。
- 已确认源：日行情、龙虎榜摘要、龙虎榜席位明细、daily basic、申万一级行业映射。
- `dragon_tiger_summary` 的来源版本为 `top_list`；席位明细来源版本为 `top_inst`。构建结果必须保留这一来源分类，不能把机构代理行伪装成完整真实席位。
- 概念点时数据、独立基准行情、公司行动及完整可交易性来源若不满足门槛，必须在质量报告中形成阻断项。

## 输出边界

输出到版本化目录 `datasets/seatflow_isd_v1/<dataset_version>/`：

- `bronze/source_inventory.json`：源文件、日期、行数和哈希清单。
- `silver/events.jsonl`、`participations.jsonl`、`cycles.jsonl`：事件、席位参与和周期事实。
- `gold/universe.jsonl`、`labels.jsonl`：每日主预测池和 1/3/5 日标签。
- `samples/samples.jsonl`：按周期日生成、去重并携带 cutoff/available-time 的模型样本元数据。
- `splits/split_definition.json`：按 `cycle_id` 原子化的滚动切分及 blind tail。
- `schemas/*.json`：节点、边、标签和样本契约。
- `manifests/dataset_manifest.json`：输入、策略、schema、split、行数和质量结果哈希。
- `quality_reports/quality_report.json`：覆盖率、PIT、标签成熟、周期原子性、席位身份范围和阻断原因。
- 只有全部正式门禁通过时才生成 `manifests/dataset_seal.json`；否则保留完整工程数据集并明确 `dataset_valid=false`。

## 架构与数据流

1. `source_adapter` 只读扫描旧快照目录，校验每个 `manifest.json` 与 `data.json` 的哈希/行数，并转换为 V3 领域记录。
2. 事件适配器按股票、交易日、上榜原因/披露批次构造稳定 `event_id`，席位明细按点时可用时间绑定事件。
3. 复用 `dataset.cycle.build_cycles`、`dataset.samples.build_samples` 和 `dataset.splits`，交易日位移一律使用真实交易日历。
4. 标签构造使用 t+1 开盘入场、t+1/t+3/t+5 收盘退出；日级路径与累计标签互相校验，尾部未成熟样本保留但不进入正式训练。
5. 复用 `dataset.materializer` 与 `dataset.seal` 计算门禁，调用方不得自报通过。
6. 写出采用临时目录加原子重命名；失败构建不得覆盖已完成版本。

## 失败处理

- 源 manifest/hash/行数不一致：立即停止该版本构建。
- `available_time > cutoff_time`、ID 进入连续特征、未成熟标签进入历史画像、Cycle 跨切分：正式 Seal 阻断。
- 数据源覆盖不足：数据行仍可作为工程产物落盘，但 `dataset_valid=false`，报告给出逐项覆盖率和缺口。
- `top_inst` 中 `institutional_flow_proxy` 与 `named_seat` 必须分开统计；任何代理席位都会阻止“完整真实 Seat 图”的正式声明。

## 近期数据增量

- 使用现有 Tushare 摄取链按交易日追加缺失快照，不覆盖 2026-07-17 及以前文件。
- 截止日固定为 2026-08-06；若当日任一核心源未完整返回，仍保留该日缺口证据并阻断正式有效状态，不静默缩短目标窗口。
- 增量结束后重新生成源 inventory，最终 `date_range`、行数和哈希只能来自实际落盘文件，不能写死为 2026-07-17。

## 测试与验收

- 单元测试先覆盖：快照发现/校验、交易日周期延长、事件-席位绑定、标签价格政策、日级路径一致性、尾部成熟状态、cycle split 原子性、确定性哈希和 fail-closed。
- 集成测试使用小型真实结构 fixture 运行完整构建并检查目录契约。
- 最终在真实 856 日输入上运行构建，重新读取输出并核对行数、哈希、日期范围和质量结论。
- 完成声明必须区分：工程数据集已产出；正式 Dataset Seal 是否通过；缺失源数据是否仍阻断正式训练。

## 范围外

- 本次不训练 Base/Seat 模型，不生成 OOF 残差，不运行 blind 指标或成本后回测。
- 本次不下载或伪造缺失的概念、公司行动、基准和可交易性数据；后续可在同一输入适配边界补充。
- 不修改 V1.0 冻结的周期、标签、切分和防泄漏口径。
