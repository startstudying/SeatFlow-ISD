# SeatFlow Terra 剩余交付 PRD

更新时间：2026-07-19  
执行边界：本文件只定义 Terra 可执行的剩余交付范围，不代表当前项目已完成模型训练、真实回测、盲测或竞赛就绪。

## 1. 当前基线与证据

### 已有能力
- Python 核心包已具备离线可运行的数据、构图、快照、只读 API 和 smoke 预测契约。
- 文档明确当前不宣称 DT-SeatHGT 训练完成，完整 HGT/PyG、真实数据审计、回测和前后端页面仍待数据门槛后完成：`README.md:16`。
- 数据源策略已区分 Tushare、AKShare、SampleProvider，且 SampleProvider 不得用于研究结论：`docs/data-sources.md:7`、`docs/data-sources.md:12`。
- `top_inst` 仅为机构席位明细，权限不足时不得伪造完整席位图：`README.md:71`、`docs/data-sources.md:27`、`docs/data-sources.md:28`。
- 图构建已有 `available_time` / `cutoff_time` 防泄漏边界：`docs/training-preflight.md:8`、`src/seatflow/graph/builder.py:30`、`src/seatflow/graph/builder.py:37`、`src/seatflow/graph/builder.py:41`。
- 快照 manifest 已包含 SHA-256、行数、质量报告元数据：`docs/data-sources.md:42`、`src/seatflow/data/store.py:57`、`src/seatflow/data/store.py:60`。
- 当前训练层只有时间切分 preflight，未实现 PyG/HGT：`docs/training-preflight.md:17`。
- 参考包中的 AI 服务、模型、Java/Next 仅能作为草图，不能当成可合并实现：`.omx/reference-init/DT-SeatHGT-Project-Init-v1.0/ai-service/app/main.py:14`、`.omx/reference-init/DT-SeatHGT-Project-Init-v1.0/ai-service/app/main.py:17`、`.omx/reference-init/DT-SeatHGT-Project-Init-v1.0/ai-service/src/models/dt_seathgt.py:1`、`.omx/reference-init/DT-SeatHGT-Project-Init-v1.0/ai-service/src/models/dt_seathgt.py:20`。

### 本次只读验证结果
- `PYTHONPATH=src python -m unittest discover -s tests -v`：24 tests OK。
- `.omx\integration-venv\Scripts\python.exe -m ruff check .`：All checks passed。
- `PYTHONPATH=src python -m seatflow preflight --config configs/default.toml`：passed=true，但 provider 为 `sample` 且 seat_detail_scope 为 `synthetic`。
- `PYTHONPATH=src .omx\integration-venv\Scripts\python.exe -m mypy src tests`：失败 3 项，均在 tests 类型标注/可空推断，不是产品代码功能失败，但应在 M0 解决。

## 2. 明确的完成口径

| 口径 | 可声明条件 | 不可替代证据 |
| --- | --- | --- |
| 工程完成 | 代码、配置、CLI/API、Docker、CI 测试全部通过，接口 fail-closed | Mock UI、HTTP 2xx、局部 smoke |
| 真实数据就绪 | 有授权 token/数据、覆盖率审计、manifest hash、字段单位/发布时间证明、缺失报告 | sample 数据、AKShare 单日成功、无权限 `top_list` |
| 模型有效 | 多时间窗、多 seed、与 MLP/LSTM/GAT/HIST/THGNN 基线比较，验证/测试集统计显著且盲测不过拟合 | 单折、单 seed、只比 smoke predictor |
| 回测有效 | 使用严格时点可得预测、交易成本/涨跌停/停牌/容量约束、滚动 out-of-sample、基准比较 | 收益曲线截图、无成交约束的理想回测 |
| 竞赛/研究就绪 | 满足比赛提交格式、盲测留痕、可复现实验包、声明边界、无泄漏审计通过 | 本地构建通过、样例提交、公开测试单次好结果 |

### 2.1 席位明细范围与允许声明矩阵

| `seat_detail_scope` | 判定依据 | 允许执行 | 允许声明 | 明确禁止 |
| --- | --- | --- | --- | --- |
| `complete_seat` | 授权数据合同明确覆盖完整 brokerage-seat，且 live audit 达到覆盖阈值 | 完整模型工程、正式训练、正式评估 | `model_validity_scope=complete_seat`，前提是其余模型/盲测门槛通过 | 仅凭 provider 静态 capability 或接口 2xx 判定完整 |
| `institutional_only` | Tushare `top_inst` 等只覆盖机构席位，且 live audit 证实权限与覆盖 | 完整工程实现、机构范围训练/回测 | `model_validity_scope=institutional_only` | 宣称完整 brokerage-seat、完整 SeatHGT 数据覆盖或将机构结果外推到全部席位 |
| `synthetic` | SampleProvider、人工 fixture | 单元测试、CPU smoke、接口降级态 | `engineering_smoke=true` | `data_ready=true`、`model_valid=true`、`backtest_valid=true` |
| `none` | 无可用席位明细 | 非席位基线、空态/错误态、数据审计 | 阻塞原因与缺口 | 任何 SeatHGT 有效性声明 |

`seat_detail_scope` 必须由真实 endpoint/source audit 产生，不能直接采用
`ProviderStatus.supports_seat_detail` 或静态配置。完整项目状态仍保持 fail-closed；机构范围成果只能作为受限研究结果单独发布。

### 2.2 执行拓扑：工程实现轨与证据晋级轨

- **工程实现轨**：允许使用确定性 fixture 实现并测试 M2-M10 的代码、契约、空态和降级态；它只能推进 `engineering_complete` 的子状态。
- **证据晋级轨**：必须依次通过真实数据审计、真实样本、预注册实验、正式训练、OOS 回测和冻结盲测，才能推进 `data_ready`、`model_valid`、`backtest_valid`、`competition_ready`。
- M1 阻塞时，Terra 不应停止全部工程工作；可以继续工程实现轨，但所有真实数据/模型/回测状态保持 false，并在 UI/API 暴露阻塞原因。
- 任一阶段的 fixture/synthetic 验收不能替代同阶段的真实证据验收。

### 2.3 数据与服务边界 ADR

- Parquet + manifest hash 是训练和审计的不可变事实源；DuckDB 作为本地分析/查询层。
- MySQL 保存业务元数据、实验索引、任务、用户和预测引用，不复制为训练事实源。
- Neo4j 仅作为可重建的图查询/可视化投影；其内容必须能从 versioned snapshot 重建，不作为标签或训练真值。
- Redis 初版仅在异步任务/缓存确有需要时启用，不得成为不可审计状态源。
- FastAPI 负责数据/模型/推理/实验产物访问；Java 负责业务编排、权限、任务状态和聚合；Next.js 只消费 Java 契约，不实现模型逻辑。

## 3. 分阶段 PRD

### M0 - 仓库卫生与执行基线

目标：在不改变产品行为的前提下，让 Terra 能从干净质量门槛开始执行。

涉及文件/模块：
- `pyproject.toml`
- `tests/test_api.py`
- `tests/test_pipeline.py`
- `.gitignore`
- `README.md`
- 新增 `scripts/quality.ps1`
- 新增 `scripts/doctor.ps1`
- 新增 `docs/environment-matrix.md`
- 将最终 PRD、测试规范、ADR、审查结论纳入版本控制；本计划通过 `.gitignore` 仅放行指定 `.omx/plans/*.md` 与 review 文件，其他 `.omx` 状态仍忽略

依赖：
- 现有 Python 3.11+，优先使用 `.omx/integration-venv` 中已存在工具；若重建环境，执行 `pip install -e ".[dev,api,data]"`。
- Terra 必须按官方兼容矩阵核验并锁定 Python、torch、torch-geometric、CPU/CUDA、MLflow、JDK、Maven、Node、Next.js、Docker Compose 版本；不能凭最新版本猜测兼容性。

验收门槛：
- `PYTHONPATH=src python -m unittest discover -s tests -v` 通过。
- `ruff check .` 通过。
- `mypy src tests` 或约定的严格范围通过；如果 tests 不纳入 mypy，则必须在文档中明确并改为 `mypy src`。
- `seatflow preflight --config configs/default.toml` 输出 passed=true，并保留 sample/synthetic 警告。
- `.gitignore` 明确排除 `data/`、`checkpoints/`、`artifacts/`、`logs/`、`.omx/*venv*`、token/secret。
- `scripts/doctor.ps1` 输出跨栈版本、GPU/CUDA 可用性、PyG 最小算子 smoke、JDK/Maven、Node/npm 和 Docker Compose 状态。
- 形成 version lock；`pyproject.toml` 增加独立 `tracking` extra 管理 MLflow，并记录 CPU fallback。
- 计划、测试规范和共识审查可被 Git 跟踪，且 `.omx` 的 venv/state/log 仍被忽略。
- 建立初始基线提交策略；每个里程碑只在门槛通过后使用 Lore 格式提交，禁止把数据、secret、大模型 artifact 纳入提交。

阻塞条件：
- 环境无法安装 dev 工具。
- torch/PyG/运行时兼容矩阵无法形成，且 CPU smoke 也不可运行。
- untracked 大文件/凭据已进入版本控制准备区。

证据：
- 测试日志、ruff/mypy 日志、preflight JSON、`git status --short`。

### M1 - 真实数据审计与权限门槛

目标：将真实数据可用性从“能调用”提升为“可训练前置证据”，并 fail-closed。

涉及文件/模块：
- `src/seatflow/data/providers.py`
- `src/seatflow/data/quality.py`
- `src/seatflow/data/store.py`
- `src/seatflow/pipeline.py`
- 新增 `src/seatflow/data/audit.py`
- 新增 `src/seatflow/data/calendar.py`
- 新增 `configs/data_audit.toml`
- 新增 `docs/data-audit-report-template.md`
- 新增 `tests/test_data_audit.py`

依赖：
- 轮换后的 `TUSHARE_TOKEN`，不得记录 token。
- 明确 `TUSHARE_ALLOW_INSECURE_HTTP=1` 时才允许 HTTP token 请求。
- 若使用 AKShare，只作为交叉校验或免费降级，不作为完整席位训练依据。

功能要求：
- 数据审计 CLI：`seatflow audit-data --provider tushare --start YYYY-MM-DD --end YYYY-MM-DD --require-seat-detail`。
- 审计输出覆盖：
  - endpoint 权限矩阵：daily/top_list/top_inst；
  - 日期覆盖率、交易日缺失率、空表比例；
  - `available_time` 证明及 source publication 假设；
  - 单位口径：成交量、成交额、买卖金额；
  - 席位明细范围：`institutional_only`、`complete_seat`、`synthetic`、`none`；
  - 数据 hash manifest 汇总；
  - 可训练/不可训练判定。
- `synthetic/none` 只能进入工程 smoke；`institutional_only` 可进入明确命名的机构范围训练，但不得产生完整席位声明；只有 `complete_seat` 才允许尝试完整席位有效性晋级。
- live audit 必须实际调用目标 endpoint；provider 静态 status 只能用于展示，不能作为权限证据。

量化晋级阈值（执行前写入 `configs/data_audit.toml` 并锁定 hash；任何调整必须形成 ADR，且不得在查看盲测结果后调整）：
- 交易日历：目标区间 100% 可解析，日期单调且无重复。
- 日线/复权因子：可交易股票-交易日覆盖率默认 >=99.5%，未解释缺失 <=0.5%。
- `top_list`：目标交易日请求覆盖率 100%；空结果必须被记录并可与第二来源或官方日历交叉解释。
- `top_inst`：对当日 `top_list` 标的请求成功率 100%；任何覆盖只能判为 `institutional_only`，除非另有明确的完整席位数据合同。
- 完整训练历史：默认 >=720 个交易日；至少支持 2 个滚动窗口，每个训练窗 >=252 日、验证/测试窗各 >=60 日。
- 样本充分性：默认每个正式训练窗 >=10,000 个合格 stock-event 样本，验证/测试各 >=2,000；每个预注册分类任务的少数类比例 >=5%。不满足时必须出具统计功效/降级 ADR，不能静默放宽。

验收门槛：
- 至少一个连续历史窗口完成审计（建议 2021-01-01 至最新可得完整交易月末；若权限不足，记录为阻塞而非降级完成）。
- `audit_report.json` 包含 `data_readiness=false|true`、`seat_detail_scope`、`coverage_by_dataset`、`leakage_assumptions`、`hashes`。
- 缺 token、缺 top_inst 权限、网络失败、字段单位未确认时均 fail-closed。

阻塞条件：
- 没有轮换后的 Tushare token。
- `top_inst` 权限不足且无另行授权的完整席位明细数据。
- 无法确认交易日历、复权、停牌/涨跌停、发布时间。

证据：
- `artifacts/data_audit/<run_id>/audit_report.json`
- 原始/处理/特征/图 manifest hash
- 采集日志脱敏版

### M2 - 标签、样本窗口与动态图张量化

目标：把已有 dataclass 图快照安全转换为可训练时序样本与 PyG `HeteroData`，并定义标签。

涉及文件/模块：
- 新增 `src/seatflow/labels.py`
- 新增 `src/seatflow/datasets/windows.py`
- 新增 `src/seatflow/datasets/pyg.py`
- 新增 `src/seatflow/graph/tensorize.py`
- 新增 `src/seatflow/training/splits.py`
- 新增 `configs/labeling.toml`
- 新增 `tests/test_labels.py`
- 新增 `tests/test_pyg_tensorize.py`
- 更新 `src/seatflow/cli.py`

依赖：
- 工程实现轨：M0 通过，可使用 deterministic fixture 实现；只能产生 engineering smoke。
- 证据晋级轨：M1 live audit 通过，且 scope/覆盖/可用时间满足对应声明门槛。
- `pyproject.toml` 的 `ml` 可选依赖已声明 torch/PyG：`pyproject.toml:18`、`pyproject.toml:19`；执行前需验证安装矩阵。

功能要求：
- 标签：
  - horizon：1/3/5 个交易日，与现有 config 一致：`configs/default.toml:14`。
  - 分类：上涨概率、超额收益方向或分位标签；回归：未来收益。
  - 标签必须晚于 feature cutoff，严格禁止未来行情入特征。
- 样本：
  - `as_of_date`、`cutoff_time`、`target_stock_id`、`label_window` 全部写入样本 manifest。
  - 支持 rolling window 和固定 train/valid/test 切分。
- 张量化：
  - 五类节点 Seat/Stock/Industry/Concept/EventReason 进入 PyG `HeteroData`。
  - 边类型包含 seat->stock buy/sell、stock->industry、stock->concept、stock->event_reason。
  - 保存 node/edge id mapping、feature schema、dtype、shape。

验收门槛：
- 单元测试证明：任意 `available_time > cutoff_time` 的记录不会进入样本张量。
- 标签生成测试证明：标签起点晚于 cutoff，不能使用同日收盘后不可得字段。
- 对真实数据产出 `dataset_manifest.json`，包含 split、shape、label distribution、missingness。
- 若真实数据未通过 M1，只允许 sample smoke，不允许产出 model-valid 声明。

阻塞条件：
- 真实交易日历/复权规则未定。
- PyTorch/PyG 安装失败或 CUDA/CPU 兼容性不满足。

证据：
- `artifacts/datasets/<run_id>/dataset_manifest.json`
- 张量 shape/mapping 快照
- 泄漏测试日志

### M2.5 - 实验协议、盲测和 lineage 前置锁定

目标：在第一次可比较训练前冻结实验规则，避免以结果反向选择 split、指标、baseline 或 seed。

涉及文件/模块：
- 新增 `configs/experiments/protocol.toml`
- 新增 `src/seatflow/evaluation/protocol.py`
- 新增 `src/seatflow/tracking/lineage.py`
- 新增 `docs/experiment-preregistration.md`
- 新增 `tests/test_experiment_protocol.py`
- 新增 `tests/test_lineage.py`

功能要求：
- 冻结内部 blind holdout 日期、train/valid/test rolling windows、purge/embargo；embargo 默认不少于最大标签 horizon（当前 5 个交易日）。
- 冻结一个主指标、次指标、baseline 列表、seed 列表、超参数搜索预算、早停规则、失败 run 处理方式和晋级阈值。
- 主结果必须报告各 seed/窗口明细、均值、标准差、最差窗口和置信区间；禁止只报告 best seed/best fold。
- 从第一次 baseline 训练开始记录 dataset/config/code/dirty-state/seed/split/checkpoint hash；MLflow 尚不可用时写本地 append-only lineage JSON。
- blind 标签和最终报告由独立命令打开；训练命令不能读取 blind 标签或修改 blind manifest。

验收门槛：
- `protocol.lock.json`、`blind_manifest.json` 和 `lineage_schema.json` 生成并校验 hash。
- 修改已冻结协议会生成新 protocol version，旧 run 不得与新协议混合汇总。
- 人工尝试随机切分、缩短 embargo、读取 blind 标签或只选择最佳 seed 时测试失败。

阻塞条件：
- 无法定义内部 blind 时间段或数据不足以同时保留训练、验证、测试和 blind。
- 主指标、baseline 或调参预算尚未冻结。

证据：
- `artifacts/protocol/<protocol_id>/protocol.lock.json`
- `artifacts/protocol/<protocol_id>/blind_manifest.json`
- 初始 `lineage.jsonl`

### M3 - 基线实验套件

目标：先建立可复现实验基线，再开发 DT-SeatHGT。

涉及文件/模块：
- 新增 `src/seatflow/models/baselines.py`
- 新增 `src/seatflow/training/runner.py`
- 新增 `src/seatflow/training/metrics.py`
- 新增 `src/seatflow/evaluation/report.py`
- 使用 M2.5 的 `src/seatflow/tracking/lineage.py`
- 新增 `configs/experiments/baselines/*.toml`
- 新增 `tests/test_metrics.py`
- 新增 `tests/test_training_runner_smoke.py`

依赖：
- 工程 smoke 依赖 M2 fixture 数据集；正式比较必须依赖 M1/M2 真实数据门槛和 M2.5 protocol lock。
- torch/PyG 安装验证。

功能要求：
- 实现并比较：
  - Naive/last-return/statistical baseline；
  - MLP；
  - LSTM/temporal sequence baseline；
  - GAT；
  - HIST/THGNN 可先作为结构化近似基线，但必须标注是否忠实复现。
- 指标：
  - AUC、PR-AUC、accuracy/F1、MSE/MAE、IC/RankIC、分 horizon 指标；
  - 每个 split、seed、window 独立报告；
  - 与 naive 和 smoke predictor 明确隔离。

验收门槛：
- 至少 3 个 seed、至少 2 个滚动窗口，valid/test 指标全部写入 `metrics.jsonl`。
- 汇总同时报告 mean/std、最差窗口、置信区间和所有失败 run；不能用 best seed/best fold 代替。
- 无真实数据时只允许 `sample_smoke=true`，不能标记 `model_valid=true`。
- 基线训练可从 CLI 复现：`seatflow train --experiment configs/experiments/baselines/mlp.toml`。

阻塞条件：
- 数据量不足，标签单一或样本极度不平衡。
- 指标无 baseline 对照或不可复现。

证据：
- `artifacts/experiments/<run_id>/metrics.jsonl`
- `artifacts/experiments/<run_id>/config.lock.toml`
- checkpoint hash 与 dataset hash lineage

### M4 - 完整 DT-SeatHGT/PyG 训练

目标：实现论文方向所需的动态异构图模型，并能在真实数据上训练、评估、解释。

涉及文件/模块：
- 新增 `src/seatflow/models/dt_seathgt.py`
- 新增 `src/seatflow/models/hgt_layers.py`
- 新增 `src/seatflow/models/heads.py`
- 新增 `src/seatflow/models/seat_profile.py`
- 新增 `src/seatflow/training/losses.py`
- 新增 `src/seatflow/training/checkpointing.py`
- 新增 `src/seatflow/explain/paths.py`
- 新增 `configs/experiments/dt_seathgt/*.toml`
- 新增 `tests/test_dt_seathgt_shapes.py`
- 新增 `tests/test_checkpoint_lineage.py`

依赖：
- M3 基线套件可复现，M2.5 protocol/lineage 已冻结。
- PyG HeteroData 数据集稳定。

功能要求：
- 模型包含：
  - HGTConv 或等价异构注意力；
  - 时间衰减/动态图窗口编码；
  - 席位软画像；
  - 关系类型感知注意力；
  - 多任务头：上涨概率、未来 1/3/5 日收益、关键席位/路径解释。
- 训练包含：
  - deterministic seed；
  - early stopping；
  - checkpoint；
  - gradient clipping；
  - CPU smoke 和 GPU 正式训练路径；
  - 模型/数据/配置 hash lineage。

验收门槛：
- shape/unit tests 覆盖所有 node/edge type 和 horizon heads。
- CPU 小样本 smoke 通过。
- 真实数据 3 seed x >=2 rolling windows 完成，不得只用单折。
- DT-SeatHGT 至少在预注册主指标上优于强 baseline；若未优于，则工程可完成但模型有效=false。
- explanation 必须通过删除扰动、子图稳定性或反事实敏感性测试；attention 权重本身不得直接表述为因果解释。
- `institutional_only` 数据只能产出同范围模型结论；完整模型命名和发布元数据必须包含 `model_validity_scope`。

阻塞条件：
- GPU 不可用不阻塞工程 smoke，但阻塞正式训练效率声明。
- 真实数据覆盖不足或 seat_detail_scope 不满足，阻塞完整模型有效声明。

证据：
- checkpoint、config lock、dataset hash、metrics、attention/path artifacts。

### M5 - MLflow 服务化与实验血缘增强

目标：把 M2.5 起已存在的最小 lineage 接入 MLflow，并增强查询、导出和服务化；不得在此阶段才开始记录正式实验。

涉及文件/模块：
- 新增 `src/seatflow/tracking/mlflow_client.py`
- 更新 `src/seatflow/tracking/lineage.py`
- 新增 `configs/mlflow.toml`
- 更新 train/evaluate/backtest CLI
- 新增 `tests/test_lineage.py`

依赖：
- M3/M4 runner 输出稳定。
- 是否使用本地 file store 或 MLflow server 需在执行时按环境选择；默认先本地 file store，Docker 阶段再服务化。

功能要求：
- 每个 run 记录：
  - git commit 或 dirty 状态；
  - dataset manifest hash；
  - config hash；
  - code package version；
  - metrics；
  - artifacts；
  - seed/split/window。
- 若 MLflow 不可用，必须 fail-closed 或降级为本地 JSON lineage，不能静默丢失。
- 导入 MLflow 的 run 必须与原始 append-only lineage hash 一致；不允许重写历史指标。

验收门槛：
- `mlruns/` 或远端 tracking 中可按 run_id 找到完整 artifact lineage。
- `seatflow export-run --run-id ...` 可生成复现包。

阻塞条件：
- MLflow 服务/文件权限不可写。
- 运行环境无法记录 Git 状态。

证据：
- MLflow run 页面或导出目录。
- `lineage.json`。

### M6 - 真实回测与盲测

目标：把模型预测转化为可审计策略表现，并与模型指标分开判断。

涉及文件/模块：
- 新增 `src/seatflow/backtest/engine.py`
- 新增 `src/seatflow/backtest/rules.py`
- 新增 `src/seatflow/backtest/costs.py`
- 新增 `src/seatflow/backtest/reports.py`
- 新增 `src/seatflow/evaluation/blind.py`
- 新增 `configs/backtest/*.toml`
- 新增 `tests/test_backtest_engine.py`
- 新增 `tests/test_blind_split.py`

依赖：
- M5 lineage 可用。
- 明确交易成本、滑点、涨跌停、停牌、容量、调仓频率。
- 盲测区间必须在调参完成前冻结。

功能要求：
- 回测：
  - 仅使用当日 cutoff 前可得预测；
  - 龙虎榜/席位信号若收盘后发布，最早在下一交易日开盘或预注册 VWAP 窗口执行，禁止用同日收盘成交；
  - 支持 top-k、阈值、多 horizon 组合；
  - 支持 A 股 T+1、100 股整数手、停牌、ST/退市/新股状态、日期化涨跌停、分市场费用、卖出印花税、滑点和容量限制；
  - 股票池必须按当时可得 master 数据构建，禁止用当前存续股票列表造成 survivorship bias；
  - 复权仅用于收益计算，成交价格与可交易性规则必须使用当时原始价格和 corporate-action 处理；
  - 输出收益、最大回撤、Sharpe、换手、命中率、暴露、容量。
- 盲测：
  - blind config hash 冻结；
  - 预测产物 hash 冻结；
  - 评估脚本单独运行；
  - 禁止盲测结果回流调参。

验收门槛：
- 至少一个滚动 OOS 回测和一个冻结盲测报告。
- 回测有效=true 只有在交易规则完整、成本约束存在、预测时点可得、基准对照存在时成立。
- 必须有成本/滑点/容量敏感性分析以及最差 OOS 窗口；不能只展示累计收益曲线。
- 竞赛就绪需另有比赛提交格式和盲测合同，不由本阶段自动推出。

阻塞条件：
- 缺真实 OOS 数据。
- 无法定义可执行交易规则。
- 盲测格式/截止时间未知。

证据：
- `artifacts/backtests/<run_id>/report.json`
- `artifacts/blind/<run_id>/blind_eval.json`
- 预测与评估 hash。

### M7 - FastAPI 推理与研究服务化

目标：把真实模型与快照服务连接，但保持 fail-closed，不能伪造预测。

涉及文件/模块：
- 更新 `src/seatflow/api.py`
- 更新 `src/seatflow/service.py`
- 新增 `src/seatflow/inference/service.py`
- 新增 `src/seatflow/inference/schemas.py`
- 新增 `src/seatflow/inference/model_registry.py`
- 新增 `tests/test_inference_api.py`

依赖：
- M4/M5 有合格 checkpoint 和 lineage。
- M6 对外展示指标已产生。

功能要求：
- API 端点：
  - `GET /health`
  - `GET /api/v1/providers/{provider}/snapshot-summary`
  - `GET /api/v1/providers/{provider}/graph/{trade_date}`
  - `GET /api/v1/models`
  - `POST /api/v1/predict`
  - `GET /api/v1/runs/{run_id}`
  - `GET /api/v1/backtests/{run_id}`
- 无模型/无 lineage/无数据时返回 404/409，不返回 mock。
- 所有预测响应包含 `model_validity_status`、`data_scope`、`lineage_hash`、`risk_flags`。

验收门槛：
- API 测试证明缺 checkpoint 不返回合成预测。
- 推理请求和训练样本 schema 一致。
- FastAPI OpenAPI schema 可生成并与 Java/Next 契约匹配。

阻塞条件：
- 没有可发布 checkpoint。
- Java/Next 依赖未定义的字段。

证据：
- API 测试日志、OpenAPI JSON、推理 smoke 日志。

### M8 - Java/Spring 业务后端

目标：提供用户、实验、预测、回测、审计查询的业务编排层，不承载训练。

涉及文件/模块：
- 新增 `backend-java/pom.xml`
- 新增 `backend-java/src/main/java/...`
- 新增 `backend-java/src/test/java/...`
- 新增 `backend-java/src/main/resources/application.yml`
- 新增 `database/mysql_schema.sql`

依赖：
- M7 OpenAPI 契约稳定。
- 选择 MySQL/Redis 是否首版必须；建议首版 MySQL + 本地缓存，Redis 作为后续可选。

功能要求：
- Spring Boot API：
  - health/readiness；
  - experiments/runs 查询；
  - predictions proxy/query；
  - backtest reports；
  - audit/status；
  - 用户/角色最小权限模型（如需要登录）。
- Java 不生成预测，不改写模型结果，只做权限、聚合、缓存和审计。

验收门槛：
- Maven test 通过。
- 契约测试对 FastAPI mock server/真实 FastAPI 均通过。
- 无 FastAPI 或无模型时业务 API fail-closed。

阻塞条件：
- 是否需要真实认证/多用户权限未定。
- 数据库选型/连接凭据不可用。

证据：
- Maven test 日志、契约测试、数据库 migration checksum。

### M9 - Next.js 四页 UI

目标：实现市场总览、资金传播图、股票预测分析、实验中心四页，所有数值来自真实 API 或明确空状态。

涉及文件/模块：
- 新增 `frontend/package.json`
- 新增 `frontend/app/*`
- 新增 `frontend/components/*`
- 新增 `frontend/lib/api.ts`
- 新增 `frontend/tests/*` 或 Playwright specs

依赖：
- M8 Java API 契约稳定；若直连 FastAPI，必须明确跳过 Java 的边界。
- UI 设计参考来自 `.omx/reference-init/.../ui-design/`，但不能复用 mock 值当真实数据。

功能要求：
- 页面：
  - 市场总览 Dashboard；
  - 资金传播图；
  - 股票预测分析；
  - 实验中心。
- 每个页面必须有：
  - loading；
  - empty；
  - error；
  - degraded/data_scope；
  - lineage/risk_flags 展示。
- 若接口返回 sample/mock/degraded，不得显示为真实预测。

验收门槛：
- TypeScript typecheck、lint、build 通过。
- Playwright 或等效 E2E 覆盖四页主要状态。
- 截图证据显示真实/缺失/降级状态区分。

阻塞条件：
- API 契约未定。
- 无真实模型 run 时不能验收真实预测页面，只能验收空/降级状态。

证据：
- `npm test/build` 日志、E2E report、截图。

### M10 - Docker 与端到端部署

目标：提供可复现本地/服务器部署，证明 FastAPI、Java、Next、DB、MLflow 能协同运行。

涉及文件/模块：
- 新增 `Dockerfile` 或 `docker/fastapi.Dockerfile`
- 新增 `backend-java/Dockerfile`
- 新增 `frontend/Dockerfile`
- 新增 `deployment/docker-compose.yml`
- 新增 `deployment/.env.example`
- 新增 `deployment/README.md`
- 新增 `scripts/e2e_smoke.ps1`

依赖：
- M7-M9 服务均可独立运行。
- 数据、checkpoint、MLflow artifacts 以 volume 挂载，禁止 bake secret/checkpoint 到镜像。

功能要求：
- compose 包含：
  - fastapi-inference；
  - backend-java；
  - frontend；
  - mysql；
  - mlflow；
  - optional nginx。
- 健康检查和 readiness 分离。
- 缺真实 secret/checkpoint 时系统启动但标记 degraded，预测接口 fail-closed。

验收门槛：
- `docker compose up --build` 成功。
- `scripts/e2e_smoke.ps1` 验证 health、API 契约、UI 页面、降级状态。
- 真实 checkpoint/数据挂载后验证预测链路和回测报告查询。

阻塞条件：
- Docker 不可用。
- 端口冲突或凭据缺失。
- 模型 artifact 缺失时不能声明完整部署，只能声明降级部署。

证据：
- compose 日志、health/readiness 输出、E2E smoke report。

### M11 - 竞赛/研究就绪包

目标：把工程、模型、回测、盲测证据整理成可审查交付包。

涉及文件/模块：
- 新增 `docs/model-card.md`
- 新增 `docs/backtest-report.md`
- 新增 `docs/competition-readiness.md`
- 新增 `artifacts/release/<version>/manifest.json`
- 新增 `scripts/package_release.ps1`

依赖：
- M6 盲测通过。
- 比赛提交格式、期限、评价指标已明确。

功能要求：
- 输出：
  - model card；
  - data card；
  - experiment report；
  - backtest report；
  - blind holdout report；
  - reproducibility manifest；
  - known limitations。
- 每个结论必须绑定证据 hash。

验收门槛：
- `competition_ready=true` 仅在提交格式、盲测、复现实验包、声明边界全部满足时出现。
- 若模型未显著优于 baseline 或回测无效，仍可工程完成，但不得宣布竞赛就绪。

阻塞条件：
- 比赛盲测格式未知。
- 不能提交/验证官方或指定盲测。

证据：
- release manifest、提交包 hash、盲测结果、复现脚本日志。

## 4. 总体依赖顺序

工程实现轨：
1. M0 仓库卫生、环境矩阵、版本化计划
2. M2 标签/张量化 fixture 实现
3. M2.5 protocol/lineage 基础设施
4. M3-M5 baseline、DT-SeatHGT、MLflow 工程实现与 CPU smoke
5. M7 FastAPI fail-closed 契约
6. M8-M10 Java、Next、Docker 的真实/空态/降级态实现

证据晋级轨：
1. M1 live 数据/权限审计并确定 `seat_detail_scope`
2. M2 真实 dataset manifest 与泄漏审计
3. M2.5 冻结 protocol、blind、seed、搜索预算和 lineage schema
4. M3 正式多 seed/rolling baseline
5. M4 正式 DT-SeatHGT 训练、消融和解释稳定性
6. M5 MLflow 导入与 lineage 核验
7. M6 OOS 回测与独立 blind evaluation
8. M7-M10 真实 checkpoint 的端到端服务验证
9. M11 竞赛/研究包

五个状态位不能按工程实现轨自动晋级。Terra 每阶段更新
`artifacts/status/phase-state.json`，字段至少包含 `engineering_complete`、`data_ready`、
`model_valid`、`backtest_valid`、`competition_ready`、`scope`、`evidence_hashes`、
`blocked_reasons` 和 `last_verified_at`。

可并行准备但不得提前验收：
- M1 数据审计可与 M2 fixture 工程并行；M1 失败只阻塞证据晋级，不阻塞可逆工程实现。
- M8 Java 项目骨架可在 M7 OpenAPI 草案后并行，但真实预测接口验收必须等 M7。
- M9 UI 空/错误/降级状态可在 M8 契约草案后并行，但真实数据页面验收必须等 M7/M8。
- M10 Docker 基础 compose 可与 M7-M9 后半并行，但真实部署验收必须等服务和 artifact 成熟。

## 5. 关键非目标

- 不把 `.omx/reference-init` 的 placeholder 应用直接复制为完成品。
- 不用 sample、mock、单日 AKShare、单折好结果替代真实数据/模型/回测证据。
- 不在仓库提交 token、真实原始数据、大 checkpoint 或 MLflow 大 artifact。
- 不在 Java 或 Next 中重新实现模型逻辑。
