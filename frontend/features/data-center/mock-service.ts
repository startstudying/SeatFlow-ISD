import type { DataAsset, DataCenterPayload, DatasetVersion, GraphEdge, GraphNode, QualityReport, SyncTask } from "./types";

const assets: DataAsset[] = [
  {
    id: "asset-raw-daily-bar",
    provider: "sample",
    dataset: "daily_bar",
    store: "raw",
    rows: 124860,
    version: "v2026.01.15.1",
    owner: "ingestion",
    updatedAt: "2026-01-15T18:15:00+08:00",
    status: "ready",
    lineageHash: "a3f2c9e5b812443f8f7d5e5b8a9d1027"
  },
  {
    id: "asset-processed-seat-event",
    provider: "sample",
    dataset: "dragon_tiger_seat_event",
    store: "processed",
    rows: 3188,
    version: "v2026.01.15.2",
    owner: "seatflow-pipeline",
    updatedAt: "2026-01-15T18:22:00+08:00",
    status: "degraded",
    lineageHash: "7b56d2339ab744ca8e20a79d2f6c6231"
  },
  {
    id: "asset-feature-stock",
    provider: "sample",
    dataset: "stock_feature",
    store: "features",
    rows: 5260,
    version: "v2026.01.15.3",
    owner: "feature-store",
    updatedAt: "2026-01-15T18:28:00+08:00",
    status: "ready",
    lineageHash: "d91d60ef6d0b456e9cfc7f4728d0c119"
  },
  {
    id: "asset-graph-hetero",
    provider: "sample",
    dataset: "hetero_graph",
    store: "graphs",
    rows: 1,
    version: "v2026.01.15.graph",
    owner: "graph-builder",
    updatedAt: "2026-01-15T18:34:00+08:00",
    status: "ready",
    lineageHash: "2ee83e4be6dc45acbbd0b0a4e6b13293"
  },
  {
    id: "asset-sync-akshare",
    provider: "akshare",
    dataset: "dragon_tiger_summary",
    store: "raw",
    rows: 0,
    version: "pending",
    owner: "upstream-sync",
    updatedAt: "2026-01-15T18:35:00+08:00",
    status: "syncing",
    lineageHash: null
  },
  {
    id: "asset-empty-audit",
    provider: "tushare",
    dataset: "seat_detail_audit",
    store: "processed",
    rows: 0,
    version: "not-created",
    owner: "audit",
    updatedAt: "2026-01-15T18:00:00+08:00",
    status: "empty",
    lineageHash: null
  }
];

const quality: QualityReport[] = [
  {
    id: "quality-daily-bar",
    provider: "sample",
    dataset: "daily_bar",
    completeness: 99.2,
    validity: 98.7,
    freshnessHours: 1.1,
    issueCount: 1,
    gate: "pass",
    checkedAt: "2026-01-15T18:36:00+08:00"
  },
  {
    id: "quality-seat-event",
    provider: "sample",
    dataset: "dragon_tiger_seat_event",
    completeness: 73.4,
    validity: 91.6,
    freshnessHours: 1.3,
    issueCount: 6,
    gate: "warn",
    checkedAt: "2026-01-15T18:36:00+08:00"
  },
  {
    id: "quality-tushare-seat",
    provider: "tushare",
    dataset: "seat_detail_audit",
    completeness: 0,
    validity: 0,
    freshnessHours: 24,
    issueCount: 4,
    gate: "block",
    checkedAt: "2026-01-15T18:36:00+08:00"
  }
];

const graphNodes: GraphNode[] = [
  { id: "seat-a", name: "机构席位 A", type: "seat", value: 82 },
  { id: "600000.SH", name: "浦发银行", type: "stock", value: 67 },
  { id: "banking", name: "银行", type: "industry", value: 58 },
  { id: "ai-compute", name: "AI 算力", type: "concept", value: 48 },
  { id: "daily-deviation", name: "日内偏离", type: "event", value: 36 }
];

const graphEdges: GraphEdge[] = [
  { source: "seat-a", target: "600000.SH", relation: "buys", weight: 0.92 },
  { source: "600000.SH", target: "banking", relation: "belongs_to", weight: 0.75 },
  { source: "600000.SH", target: "daily-deviation", relation: "triggered_by", weight: 0.44 },
  { source: "seat-a", target: "ai-compute", relation: "co_moves", weight: 0.28 }
];

const versions: DatasetVersion[] = [
  {
    id: "ver-graph-20260115",
    dataset: "hetero_graph",
    version: "v2026.01.15.graph",
    tag: "current",
    rows: 1,
    createdAt: "2026-01-15T18:34:00+08:00",
    checksum: "2ee83e4be6dc45ac",
    author: "graph-builder"
  },
  {
    id: "ver-feature-20260115",
    dataset: "stock_feature",
    version: "v2026.01.15.3",
    tag: "feature-freeze",
    rows: 5260,
    createdAt: "2026-01-15T18:28:00+08:00",
    checksum: "d91d60ef6d0b456e",
    author: "feature-store"
  },
  {
    id: "ver-bar-20260114",
    dataset: "daily_bar",
    version: "v2026.01.14.1",
    tag: "previous",
    rows: 123990,
    createdAt: "2026-01-14T18:18:00+08:00",
    checksum: "9a6c04da1ebd4899",
    author: "ingestion"
  }
];

const syncTasks: SyncTask[] = [
  {
    id: "sync-akshare-001",
    source: "akshare",
    target: "raw.dragon_tiger_summary",
    status: "running",
    progress: 64,
    startedAt: "2026-01-15T18:31:00+08:00",
    message: "等待上游分批返回龙虎榜汇总"
  },
  {
    id: "sync-lineage-002",
    source: "lineage-check",
    target: "graphs.hetero_graph",
    status: "success",
    progress: 100,
    startedAt: "2026-01-15T18:29:00+08:00",
    message: "图快照 checksum 与 manifest 一致"
  },
  {
    id: "sync-seat-003",
    source: "licensed-seat",
    target: "processed.seat_detail_audit",
    status: "failed",
    progress: 18,
    startedAt: "2026-01-15T18:10:00+08:00",
    message: "缺少完整席位授权，保持 fail-closed"
  }
];

export async function fetchMockDataCenter(): Promise<DataCenterPayload> {
  await new Promise((resolve) => window.setTimeout(resolve, 220));
  const rowCount = assets.reduce((total, asset) => total + asset.rows, 0);

  return {
    mode: "mock",
    overview: {
      generatedAt: "2026-01-15T18:36:00+08:00",
      provider: "sample",
      tradeDate: "2026-01-15",
      assetCount: assets.length,
      rowCount,
      graphNodeCount: graphNodes.length,
      graphEdgeCount: graphEdges.length,
      qualityScore: 87.6,
      blockedIssues: quality.filter((item) => item.gate === "block").length,
      activeTasks: syncTasks.filter((task) => task.status === "queued" || task.status === "running").length
    },
    assets,
    quality,
    graph: {
      nodes: graphNodes,
      edges: graphEdges
    },
    versions,
    syncTasks
  };
}
