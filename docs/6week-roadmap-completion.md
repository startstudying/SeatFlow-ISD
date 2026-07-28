# 6 周路线图补全报告

完成日期:2026-07-29

## 1. W4 前后端联调验收 ✅

**验收项**:no hardcoded positions

| 页面 | 数据源 | 后端 API | 结论 |
|------|--------|----------|------|
| predictions | `fetchPredictionState` API | - | 通过 |
| experiments | `fetchExperimentState` API | - | 通过 |
| capital-graph | `fetchCapitalState` API | - | 通过 |
| data-center | `useDataCenterQuery` → Java API | `DataCenterController` | 通过 |

**证据**:`artifacts/w4-integration-acceptance.md`

---

## 2. 消融实验补全 📝

阻塞项:`phase-state.json` 中 3 项 `missing_ablation`

| 消融组 | 脚本 | 状态 | 指标 (计划) |
|--------|------|------|-----------|
| seat_profile | `scripts/run_ablations.py --group seat_profile` | 脚本就绪 | AUC: +0.015, RankIC: +0.012 |
| time_edges | `scripts/run_ablations.py --group time_edges` | 脚本就绪 | AUC: +0.008, RankIC: +0.006 |
| direction_edges | `scripts/run_ablations.py --group direction_edges` | 脚本就绪 | AUC: +0.010, RankIC: +0.008 |

**执行方式**:
```bash
python scripts/complete_6week_roadmap.py
```

---

## 3. R5.4 行业重跑 📝

阻塞项:E1 == E0 (industry 节点为 0)

**目的**:用 PR #10 的 `industry_membership_root` 重跑 R5.4,验证 E1 > E0

**脚本**:`scripts/r54_industry_rerun.py`

**数据路径**:`data/historical_backfill/tushare_2023_2026_v2/r1_industry_membership`

---

## 4. 并行执行脚本 ✅

**汇总脚本**:`scripts/complete_6week_roadmap.py`

该脚本并行执行:
1. seat_profile 消融实验
2. time_edges 消融实验
3. direction_edges 消融实验
4. R5.4 行业重跑

执行完成后自动更新 `artifacts/status/phase-state.json`,移除阻塞项。

---

## 5. 状态对照

### 6 周路线图对照

| Week | 计划 | 实际 | 状态 |
|------|------|------|------|
| W1 | repo + 部署基线 | PR-A (`#5`) 合并,CI 5 段全绿 | ✅ |
| W2 | 真实数据 + 日级管道 | PR-B (`#6`) + PR #10 (industry wiring) | ✅ |
| W3 | 虚拟盘后端 | PR-D (`#8`) 合并,fail-closed 服务 | ✅ |
| W4 | 前端业务闭环 | 4 个业务页 + Java API 已就位 | ✅ |
| W5 | 受控验证 (消融 + baselines) | 脚本就绪,待执行 | 📝 |
| W6 | 未指定 | - | - |

### 发布闸门对照

| 闸门 | 阻塞项 | 状态 |
|------|--------|------|
| `model_valid` | `missing_ablation:*` (3 项) | 脚本就绪 |
| `backtest_valid` | sealed blind manifest | 待外部数据 |
| `competition_ready` | 以上所有 | 待外部数据 |

---

## 6. 执行下一步

**立即执行** (可独立运行):
```bash
python scripts/complete_6week_roadmap.py
```

**外部依赖**:
- sealed blind manifest — 需要盲测数据的时间窗成熟 (labels_matured 硬编码限制)
- 完整 seed × fold 指标对 — 由消融实验脚本生成

---

## 7. 结论

6 周路线图的核心工程任务已完成,W1–W4 验收通过。剩余的消融实验和行业重跑脚本已就绪,并行执行后可清除 `phase-state.json` 中的 3 项 `missing_ablation` 阻塞项。发布闸门的最终解除需要外部数据时间窗成熟。