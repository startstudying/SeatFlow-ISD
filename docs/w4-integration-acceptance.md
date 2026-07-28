# W4 前后端联调验收报告

验收日期:2026-07-29

## 验收项:no hardcoded positions

| 页面 | 数据源 | Hardcode 检查 | 后端 API | 结论 |
|------|--------|---------------|----------|------|
| predictions | `fetchPredictionState` API + `maybeStubState` | 无硬编码 | - | 通过 |
| experiments | `fetchExperimentState` API + `maybeStubState` | 无硬编码 | - | 通过 |
| capital-graph | `fetchCapitalState` API + `StateFrame` | 无硬编码 | - | 通过 |
| data-center | `useDataCenterQuery` → Java API 适配 | 无硬编码 | `DataCenterController` | 通过 |

## 后端 API 路径

- `/api/v1/data-center/overview` — 数据总览
- `/api/v1/providers/{provider}/snapshot-summary` — 快照摘要
- `/api/v1/providers/{provider}/graph/{tradeDate}` — 图谱快照

## 环境变量

- `NEXT_PUBLIC_SEATFLOW_DATA_CENTER_SOURCE` — "java-api" / mock
- `NEXT_PUBLIC_SEATFLOW_BACKEND_BASE_URL` — 后端基址

## 验收结论

通过。4 个业务页面无 hardcode positions,前端通过真实/模拟双模式 API 读取数据,后端 Java 控制器已就位。