# SeatFlow / DT-SeatHGT

SeatFlow 是“DragonTiger-Agent 量化交易研究平台”方向 3 的工程项目。它将龙虎榜席位、股票、行业、概念和上榜原因组织为动态异构图，为未来 1、3、5 个交易日的收益预测与资金传播解释提供可追溯的数据入口。

## 已完成能力

- 五类节点：`Seat`、`Stock`、`Industry`、`Concept`、`EventReason`
- 席位到股票的有向交易边，以及股票到行业/概念/事件的关系边
- 指数时间衰减权重 `w = w0 * exp(-lambda * delta_days)`
- Tushare Pro REST 接入，Token 仅在运行时从环境变量读取
- AKShare 免费降级适配器和确定性 `sample` 离线数据源
- 原始、处理后、特征和图快照的 SHA-256 manifest
- `available_time` 与时间切分的防泄漏预检
- 不依赖第三方包的合成数据 smoke 流程与单元测试

## 当前交付状态

工程已具备可验证的 DT-SeatHGT/PyG CPU shape smoke、训练谱系、严格数据审计、盲测协议、真实回测门禁、FastAPI、Java 编排层、四页 Next.js UI、Docker Compose 与竞赛发布门禁。前端页面覆盖市场总览、资金传播图、预测分析和实验中心；任何缺 checkpoint、lineage、真实图快照或风险证据的响应都会显示空态、错误态或降级态，绝不伪造预测。

当前**不**宣称真实数据就绪、模型有效、真实 OOS 回测有效或竞赛就绪。原因是尚无经过审计的历史完整席位明细、轮换后的数据凭据、冻结盲测结果及比赛提交合同。使用 `seatflow phase-status` 可读取该 fail-closed 状态；只有证据哈希齐全后才允许升级状态。

## 快速开始

要求 Python 3.11+。

```powershell
$env:PYTHONPATH = "src"
python -m seatflow validate-config --config configs/default.toml
python -m seatflow preflight --config configs/default.toml
python -m seatflow smoke --config configs/default.toml
python -m unittest discover -s tests -v
```

如需安装为可编辑包：

```powershell
python -m pip install -e . --no-deps
seatflow smoke --config configs/default.toml
```

机器学习和免费数据源依赖仅做可选声明，项目不会自动下载或安装：

```powershell
python -m pip install -e ".[ml,tracking,api,dev,data]"
```

## 数据接入

### 默认离线验证

`sample` 提供确定性的合成数据，仅用于验证采集、质量检查、特征和图快照流程。

```powershell
$env:PYTHONPATH = "src"
python -m seatflow ingest --provider sample --trade-date 2026-01-15
python -m seatflow build-artifacts --provider sample --trade-date 2026-01-15
```

### Tushare Pro

Token 只从当前进程的 `TUSHARE_TOKEN` 环境变量读取，不会写入配置、日志或产物。请使用新生成的 Token，不要复用已公开的 Token。

项目根目录的 `.env`（已被 Git 忽略）会在 `seatflow` CLI 启动时自动读取 `TUSHARE_*` 与 `SEATFLOW_*` 设置；系统环境变量优先于该文件。

```powershell
$env:TUSHARE_TOKEN = "<new-token>"
$env:TUSHARE_ALLOW_INSECURE_HTTP = "1"  # Official REST documentation uses HTTP.
python -m seatflow provider-status --provider tushare
python -m seatflow ingest --provider tushare --trade-date 2026-07-17
python -m seatflow build-artifacts --provider tushare --trade-date 2026-07-17
```

接入策略：

- `daily`：日线行情；
- `top_list`：龙虎榜股票级汇总；
- `top_inst`：机构席位明细，权限不足时会明确降级，不会伪造席位图；
- 构图时依据 `available_time` 和截止时间筛选边，防止未来数据进入图快照。

### 免费降级源

安装可选 AKShare 依赖后，可获取单股票日线及龙虎榜汇总：

```powershell
python -m pip install -e ".[data]"
python -m seatflow ingest --provider akshare --stock-code 000001.SZ --trade-date 2026-07-17
```

AKShare 的公开网页源可能改版，且当前适配器不宣称提供完整席位级龙虎榜明细；它适合作为免费研究源、缓存和交叉校验，不能替代已验证的席位明细数据。

## 只读 API

安装 API 可选依赖后，可把已生成的快照提供给前端或调试工具。服务不会触发采集、训练或伪造预测。

```powershell
python -m pip install -e ".[api]"
python -m seatflow serve-api --provider sample --host 127.0.0.1 --port 8000
```

端点：`GET /health`、`GET /api/v1/providers/{provider}/snapshot-summary`、`GET /api/v1/providers/{provider}/graph/{trade_date}`。缺失快照返回 `404`。

## 项目结构

```text
SeatFlow/
├─ configs/                 # 可复现实验与时间切分配置
├─ data/                    # 本地原始/处理/特征/图数据，不提交仓库
├─ docs/                    # 架构、数据来源、契约与训练门槛
├─ src/seatflow/
│  ├─ data/                 # 领域记录、提供方、质量检查和快照存储
│  ├─ graph/                # 动态异构图类型、衰减与构建
│  ├─ training/             # 时间切分与训练前置检查
│  ├─ features.py           # 无未来信息的基础日线特征
│  ├─ pipeline.py           # 采集、规范化、快照与构图流程
│  ├─ services/             # 预测输出与 smoke 基线
│  ├─ cli.py                # 验证、采集、构图和预检入口
│  └─ config.py             # TOML 配置加载与校验
└─ tests/                   # 标准库 unittest 回归测试
```

## 后续里程碑

1. 用实际权限采集并审计历史龙虎榜、OHLCV、行业/概念和事件数据。
2. 固定按时间滚动的训练/验证/测试切分，确认标签和可交易性规则。
3. 依次实现并比较 MLP、LSTM、GAT、HIST、THGNN 基线。
4. 实现席位软画像、关系类型感知注意力和 DT-SeatHGT 多任务头。
5. 用真实实验结果接入预测、传播路径、回测与前端展示。

详细边界见 `docs/architecture.md`、`docs/data-contract.md`、`docs/data-sources.md` 和 `docs/training-preflight.md`。
