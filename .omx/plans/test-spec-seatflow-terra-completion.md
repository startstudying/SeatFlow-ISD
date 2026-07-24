# SeatFlow Terra 独立测试规范大纲

更新时间：2026-07-19  
目标：为 Terra 执行阶段提供独立、可复验、分层测试规范；本规范不依赖实现者自证。

## 1. 测试原则

1. 工程可运行、模型有效、回测有效、竞赛就绪分开判定。
2. 所有真实数据路径必须有 manifest hash、质量报告和 `available_time` 证据。
3. sample/mock 只能通过 smoke，不得晋级为模型或回测证据。
4. 每个阶段先测 fail-closed，再测成功路径。
5. 盲测配置和预测产物必须 hash 冻结，禁止盲测后调参。
6. 工程实现轨与证据晋级轨分开验收；fixture 只能证明工程行为。
7. `seat_detail_scope` 必须来自 live audit，并贯穿 dataset、checkpoint、API、UI 和 release manifest。

## 2. 基线质量门槛

命令：
- `PYTHONPATH=src python -m unittest discover -s tests -v`
- `ruff check .`
- `mypy src tests`（或文档化的 `mypy src`，但需单独记录 tests 类型债）
- `PYTHONPATH=src python -m seatflow preflight --config configs/default.toml`
- `python -m compileall src tests`
- `python -m build` + wheel install smoke（发布前）
- `powershell -File scripts/doctor.ps1`

通过标准：
- unittest 全部通过。
- ruff 无错误。
- mypy 策略稳定且无未解释失败。
- preflight passed=true，但 sample/synthetic 警告不得被误报为真实数据就绪。
- doctor 输出 Python/torch/PyG/CPU-CUDA/MLflow/JDK/Maven/Node/Next/Docker Compose 版本矩阵，PyG CPU 最小算子 smoke 必须通过。
- 最终 PRD、测试规范和共识审查被 `.gitignore` 精确放行，可由 Git 跟踪；其他 `.omx` 运行状态仍忽略。

当前发现：
- 全局 Python 未设置 `PYTHONPATH` 时 unittest 无法 import `seatflow`，执行规范必须显式设置环境或 editable install。
- `.omx/integration-venv` 中 ruff 可用；`.omx/venv` 中缺 ruff/mypy。
- 当前 `mypy src tests` 有 3 个 tests 类型错误，M0 必须修复或重新定义 mypy 范围。

## 3. 数据审计测试

覆盖：
- 缺 token：Tushare provider 报 `ProviderNotConfiguredError`，错误消息不含 token。
- 未设置 `TUSHARE_ALLOW_INSECURE_HTTP=1`：拒绝 HTTP token 请求。
- `top_inst` 权限失败：脱敏错误，`data_readiness=false`。
- AKShare 网络/字段失败：转换为 provider error，不能写入假席位。
- `--require-seat-detail` 缺席位明细时失败。
- manifest SHA-256 与 data.json 内容一致。
- 每个 dataset 的 row_count、missingness、available_time、source_version 可审计。
- provider 静态 `supports_seat_detail` 不得使 live audit 自动通过。
- `complete_seat`、`institutional_only`、`synthetic`、`none` 四种 scope 映射到正确允许声明。
- 默认阈值：calendar=100%，daily/adj>=99.5%，`top_list` 请求日覆盖=100%，`top_inst` 对目标股票请求成功=100%，训练历史>=720 交易日；阈值变更必须产生新 audit config hash。

测试文件建议：
- `tests/test_data_audit.py`
- `tests/test_provider_permissions.py`
- `tests/test_manifest_integrity.py`

验收数据：
- mock provider fixture 模拟权限、空表、字段缺失、重复席位、晚发布。
- 若有真实 token，真实采集测试必须在 integration 标记下运行，不进入默认单元测试。

## 4. 标签与泄漏测试

覆盖：
- 标签窗口严格晚于 feature cutoff。
- 停牌/涨跌停/复权规则在配置缺失时 fail-closed。
- `available_time > cutoff_time` 的记录不进入特征、边、标签。
- 随机打散跨时间 split 被拒绝。
- 历史胜率/历史收益聚合只使用 cutoff 前数据。

测试文件建议：
- `tests/test_labels.py`
- `tests/test_time_leakage.py`
- `tests/test_window_dataset.py`

通过标准：
- 对人工构造未来泄漏 fixture 必须失败。
- 每个样本 manifest 包含 cutoff、label horizon、source hash。
- rolling split 对重叠标签使用 purge/embargo，embargo 不少于最大 horizon。

## 4.1 实验协议与盲测前置测试

覆盖：
- 第一次 baseline 前已冻结 blind 时间段、主/次指标、baseline 列表、seed、搜索预算、早停和失败 run 处理规则。
- `protocol.lock.json`、`blind_manifest.json`、`lineage_schema.json` hash 可验证。
- 训练进程不能读取 blind 标签；blind evaluator 不能修改训练配置。
- 修改冻结协议会生成新 protocol version，旧/新结果不能混合汇总。
- 汇总器拒绝只选择 best seed/best fold，必须包含所有 run、mean/std、最差窗口和置信区间。

测试文件建议：
- `tests/test_experiment_protocol.py`
- `tests/test_blind_isolation.py`
- `tests/test_lineage.py`

通过标准：
- 人工尝试随机切分、缩短 embargo、读取 blind 标签、回写调参或丢弃失败 seed 时测试失败。

## 5. PyG 张量化测试

覆盖：
- 五类节点映射稳定且可复现。
- 每种 edge type shape、dtype、index bounds 正确。
- 缺失节点特征使用明确 mask，不静默填充为真实值。
- `HeteroData` round-trip 保存/加载后 shape 和 mapping hash 不变。

测试文件建议：
- `tests/test_pyg_tensorize.py`
- `tests/test_dataset_manifest.py`

通过标准：
- CPU 环境可跑最小 fixture。
- PyG 缺失时跳过正式张量化测试但保留清晰 skip reason；工程发布环境不得跳过。

## 6. 训练与模型测试

覆盖：
- baseline runner CLI 可复现。
- MLP/LSTM/GAT/HIST/THGNN baseline shape 与 metrics 输出一致。
- DT-SeatHGT forward 输出：
  - `rise_probability`；
  - 1/3/5 日收益；
  - key seats/path evidence；
  - risk flags。
- loss、early stopping、checkpoint restore、seed determinism。
- CPU smoke 与 GPU 正式训练分开。

测试文件建议：
- `tests/test_training_runner_smoke.py`
- `tests/test_metrics.py`
- `tests/test_dt_seathgt_shapes.py`
- `tests/test_checkpoint_lineage.py`

通过标准：
- 小 fixture 训练 1-2 epoch 能完成并产出 metrics/checkpoint。
- restore 后同一 batch 预测一致。
- 真实训练有效性只由多 seed、多 rolling windows 指标决定。

## 7. MLflow/Lineage 测试

覆盖：
- run_id、config hash、dataset hash、git status、checkpoint hash 全部记录。
- 第一次 baseline/DT 训练即写本地 append-only lineage；MLflow 只是后续服务化，不能成为首次记录点。
- MLflow 不可用时 fail-closed 或显式本地 lineage fallback。
- `export-run` 能复现必要文件，不包含 token/原始大数据。

测试文件建议：
- `tests/test_lineage.py`
- `tests/test_mlflow_tracking.py`

通过标准：
- 任一 metrics 行能追溯到 dataset manifest 与 code/config。

## 8. 回测与盲测测试

覆盖：
- 预测时点可得，不使用真实未来收益。
- 收盘后龙虎榜信号不能在同日收盘成交，只能在下一交易日预注册开盘/VWAP 窗口执行。
- A 股 T+1、100 股整数手、交易成本、卖出印花税、滑点、涨跌停、停牌、ST/退市/新股、容量限制生效。
- 股票池按历史 as-of master 构建，当前存续列表不能造成 survivorship bias。
- 复权收益与原始成交价格/corporate-action 处理边界正确。
- top-k/threshold 策略对空信号、无成交日、停牌日处理正确。
- blind config hash 冻结后不能被训练 CLI 修改。
- 盲测结果不能自动覆盖 tuning config。

测试文件建议：
- `tests/test_backtest_engine.py`
- `tests/test_costs_limits.py`
- `tests/test_blind_split.py`

通过标准：
- 构造型 fixture 的成交和收益可手算验证。
- 真实回测报告必须含 benchmark、turnover、drawdown、capacity、cost。
- 真实回测还必须含成本/滑点/容量敏感性和最差 OOS 窗口。

## 9. API 契约测试

覆盖：
- 现有只读快照端点保持兼容。
- 缺 provider/graph/checkpoint/model lineage 返回 404/409，不返回合成预测。
- `/api/v1/predict` 输入 schema 与训练样本 schema 一致。
- 响应包含 `model_validity_status`、`data_scope`、`lineage_hash`、`risk_flags`。
- `institutional_only` checkpoint 的 API/UI 不得展示为 complete-seat 模型。

测试文件建议：
- `tests/test_api.py` 扩展
- `tests/test_inference_api.py`
- OpenAPI schema diff test

通过标准：
- FastAPI TestClient 和真实 uvicorn smoke 均通过。

## 10. Java/Spring 测试

覆盖：
- Controller/Service/Repository 分层单元测试。
- FastAPI client 契约测试。
- 数据库 migration checksum。
- FastAPI 不可用、模型不可用、降级数据时 Java fail-closed。

命令：
- `mvn test`
- `mvn verify`
- 契约测试：Java against FastAPI mock + real FastAPI smoke。

通过标准：
- 不生成假预测。
- 业务 API 所有数值可追溯到 FastAPI/MLflow/DB。

## 11. Next.js UI 测试

覆盖：
- 四页：市场总览、资金传播图、股票预测分析、实验中心。
- loading/empty/error/degraded/real data 状态。
- API schema 类型生成或手写类型一致。
- 禁止 mock 值在 production build 默认显示。

命令：
- `npm run lint`
- `npm run typecheck`
- `npm run test`
- `npm run build`
- `npx playwright test`

通过标准：
- E2E 截图显示降级和真实状态可区分。

## 12. Docker/E2E 测试

覆盖：
- `docker compose up --build`。
- health/readiness。
- FastAPI -> Java -> Next 链路。
- MLflow artifact 可读。
- checkpoint/data volume 挂载。
- 缺 secret/checkpoint 的 degraded 启动。
- Parquet/manifest 为训练事实源；MySQL 仅业务元数据；Neo4j 投影可从 snapshot 重建。

命令：
- `docker compose -f deployment/docker-compose.yml up --build`
- `scripts/e2e_smoke.ps1`

通过标准：
- 无真实 checkpoint 时预测 fail-closed。
- 有真实 checkpoint/data 时可查询预测、run、backtest。
- `artifacts/status/phase-state.json` 的五个状态位与可验证 evidence hash 一致。

## 13. 发布/竞赛证据测试

覆盖：
- release manifest hash。
- model card/data card/backtest report/blind report 存在。
- 竞赛提交格式验证。
- 重跑脚本在新环境中能恢复评估。

通过标准：
- `competition_ready=true` 只在提交合同、盲测、复现、声明边界全部满足时出现。
