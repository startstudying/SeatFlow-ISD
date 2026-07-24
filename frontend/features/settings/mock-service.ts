import type { SettingsPayload } from "./types";

const now = "2026-07-19T09:30:15+08:00";

/** Development-only fixture. Production uses the Java adapter. */
export async function fetchMockSettings(): Promise<SettingsPayload> {
  return {
    mode: "mock",
    settings: [{
      key: "general",
      category: "general",
      value: { systemName: "SeatFlow 科研量化平台", timezone: "Asia/Shanghai", market: "A 股（中国内地）", retentionDays: 180 },
      version: 3
    }],
    dataSources: [
      { id: "tushare", displayName: "Tushare 行情数据", provider: "tushare", endpoint: "https://api.tushare.pro", metadata: { schedule: "交易日 18:00", coverage: "2023-01-01 至今" }, apiKeyMasked: "已配置（仅显示状态）", apiKeyKeyId: "key-tushare-01", version: 2 },
      { id: "akshare", displayName: "AKShare 公共数据", provider: "akshare", endpoint: "https://akshare.akfamily.xyz", metadata: { schedule: "交易日 20:00", coverage: "补充数据" }, apiKeyMasked: "无需密钥", apiKeyKeyId: null, version: 1 }
    ],
    models: [{ id: "seat-graph-model", displayName: "席位异构图模型", provider: "python-service", status: "待验证", parameters: { hiddenDim: 256, heads: 8, dropout: 0.2 }, version: 1 }],
    experiments: [{ id: "research-default", name: "研究基线实验", status: "待数据审计", config: { seed: 42, trainRatio: 0.7, validationRatio: 0.15 }, version: 1 }],
    services: [
      { serviceId: "frontend", status: "运行中", checkedAt: now, metrics: { latencyMs: 24, memory: "318MB" }, version: 1 },
      { serviceId: "backend-java", status: "运行中", checkedAt: now, metrics: { latencyMs: 36, memory: "642MB" }, version: 1 },
      { serviceId: "fastapi", status: "待验证", checkedAt: now, metrics: { gpu: "未声明" }, version: 1 }
    ],
    users: [
      { id: "researcher", displayName: "量化研究员", active: true, roles: ["RESEARCHER"], version: 1 },
      { id: "operator", displayName: "运维工程师", active: true, roles: ["OPERATOR"], version: 1 }
    ],
    roles: [
      { code: "ADMIN", name: "系统管理员", permissions: ["settings.write", "sync.execute", "audit.read"], version: 1 },
      { code: "RESEARCHER", name: "量化研究员", permissions: ["data.read", "experiment.write"], version: 1 }
    ],
    permissions: [
      { code: "settings.write", name: "修改系统设置", resource: "settings", version: 1 },
      { code: "sync.execute", name: "执行同步任务", resource: "sync", version: 1 }
    ],
    auditLogs: [
      { id: 1, actor: "researcher", action: "UPSERT", resourceType: "MODEL", resourceId: "seat-graph-model", requestId: "mock-request-001", idempotencyKey: "mock-idempotency-001", detail: { status: "待验证" }, createdAt: now },
      { id: 2, actor: "operator", action: "UPSERT", resourceType: "DATA_SOURCE", resourceId: "tushare", requestId: "mock-request-002", idempotencyKey: "mock-idempotency-002", detail: { provider: "tushare" }, createdAt: "2026-07-19T09:12:10+08:00" }
    ]
  };
}
