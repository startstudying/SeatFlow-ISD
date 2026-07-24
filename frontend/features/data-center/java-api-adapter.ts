import { backendGet, type PageResult } from "@/lib/backend-client";
import type {
  DataAsset,
  DataCenterPayload,
  DatasetVersion,
  GraphEdge,
  GraphNode,
  QualityReport,
  SyncTask
} from "./types";

type JavaAsset = {
  id: string;
  provider: string;
  dataset: string;
  store: string;
  dataScope: string;
  seatDetailScope: string;
  lineageHash: string | null;
  rowCount: number;
  version: number;
};

type JavaQuality = {
  id: string;
  provider: string;
  dataset: string;
  tradeDate: string;
  passed: boolean;
  completenessRate: number;
  duplicateRate: number;
  riskFlags: string[];
  version: number;
};

type JavaGraphSnapshot = {
  id: string;
  provider: string;
  tradeDate: string;
  nodeCount: number;
  edgeCount: number;
  dataScope: string;
  seatDetailScope: string;
  lineageHash: string | null;
  version: number;
};

type JavaDatasetVersion = {
  id: string;
  provider: string;
  dataset: string;
  versionTag: string;
  artifactUri: string;
  contentSha256: string | null;
  active: boolean;
  version: number;
};

type JavaSyncTask = {
  id: string;
  provider: string;
  dataset: string;
  mode: string;
  status: string;
  progress: number;
  message: string;
  riskFlags: string[];
  version: number;
  createdAt: string;
  updatedAt: string;
};

type JavaGraphDetail = {
  graph?: {
    nodes?: Array<Record<string, unknown>>;
    edges?: Array<Record<string, unknown>>;
  };
};

const provider = process.env.NEXT_PUBLIC_SEATFLOW_PROVIDER ?? "sample";
const tradeDate = process.env.NEXT_PUBLIC_SEATFLOW_TRADE_DATE ?? "2026-01-15";
const pageSize = 100;

/**
 * The browser only calls the Spring Boot gateway.  Graph details are also
 * requested through that gateway, never from FastAPI directly.
 */
export async function fetchJavaDataCenter(): Promise<DataCenterPayload> {
  const [assetsPage, qualityPage, graphPage, versionsPage, taskPage] = await Promise.all([
    backendGet<PageResult<JavaAsset>>(`/api/v1/data-assets?page=0&size=${pageSize}`),
    backendGet<PageResult<JavaQuality>>(`/api/v1/data-quality/reports?page=0&size=${pageSize}`),
    backendGet<PageResult<JavaGraphSnapshot>>(`/api/v1/graph/snapshots?page=0&size=${pageSize}`),
    backendGet<PageResult<JavaDatasetVersion>>(`/api/v1/datasets/versions?page=0&size=${pageSize}`),
    backendGet<PageResult<JavaSyncTask>>(`/api/v1/sync/tasks?page=0&size=${pageSize}`)
  ]);

  const assets = assetsPage.items.map(toAsset);
  const quality = qualityPage.items.map(toQuality);
  const graph = await loadGraphDetail(graphPage.items);
  const versions = versionsPage.items.map(toVersion);
  const syncTasks = taskPage.items.map(toTask);

  return {
    mode: "java-api",
    overview: {
      generatedAt: new Date().toISOString(),
      provider,
      tradeDate,
      assetCount: assetsPage.total,
      rowCount: assets.reduce((total, asset) => total + asset.rows, 0),
      graphNodeCount: graph.nodes.length || graphPage.items.reduce((total, snapshot) => total + snapshot.nodeCount, 0),
      graphEdgeCount: graph.edges.length || graphPage.items.reduce((total, snapshot) => total + snapshot.edgeCount, 0),
      qualityScore: averageQuality(quality),
      blockedIssues: quality.filter((item) => item.gate === "block").length,
      activeTasks: syncTasks.filter((task) => task.status === "queued" || task.status === "running").length
    },
    assets,
    quality,
    graph,
    versions,
    syncTasks
  };
}

function toAsset(value: JavaAsset): DataAsset {
  return {
    id: value.id,
    provider: value.provider,
    dataset: value.dataset,
    store: safeStore(value.store),
    rows: value.rowCount,
    version: `v${value.version}`,
    owner: value.dataScope || "数据服务",
    updatedAt: new Date().toISOString(),
    status: value.rowCount > 0 && value.lineageHash ? "ready" : value.rowCount > 0 ? "degraded" : "empty",
    lineageHash: value.lineageHash
  };
}

function toQuality(value: JavaQuality): QualityReport {
  const completeness = percentage(value.completenessRate);
  const validity = Math.max(0, 100 - percentage(value.duplicateRate));
  return {
    id: value.id,
    provider: value.provider,
    dataset: value.dataset,
    completeness,
    validity,
    freshnessHours: 0,
    issueCount: value.riskFlags.length,
    gate: value.passed ? "pass" : value.riskFlags.length ? "block" : "warn",
    checkedAt: `${value.tradeDate}T00:00:00+08:00`
  };
}

function toVersion(value: JavaDatasetVersion): DatasetVersion {
  return {
    id: value.id,
    dataset: value.dataset,
    version: value.versionTag,
    tag: value.active ? "当前版本" : "历史版本",
    rows: 0,
    createdAt: new Date().toISOString(),
    checksum: value.contentSha256 ?? "未登记",
    author: value.provider
  };
}

function toTask(value: JavaSyncTask): SyncTask {
  return {
    id: value.id,
    source: value.provider,
    target: `${value.dataset} / ${value.mode}`,
    status: safeTaskStatus(value.status),
    progress: value.progress,
    startedAt: value.createdAt,
    message: value.riskFlags.length ? `${value.message}（${value.riskFlags.join(", ")}）` : value.message
  };
}

async function loadGraphDetail(snapshots: JavaGraphSnapshot[]): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> {
  const snapshot = snapshots.find((item) => item.provider === provider && item.tradeDate === tradeDate) ?? snapshots[0];
  if (!snapshot) {
    return { nodes: [], edges: [] };
  }

  try {
    const detail = await backendGet<JavaGraphDetail>(`/api/v1/providers/${encodeURIComponent(snapshot.provider)}/graph/${snapshot.tradeDate}`);
    return graphFromDetail(detail);
  } catch {
    // Snapshot metadata is kept visible in overview; no fabricated graph is rendered.
    return { nodes: [], edges: [] };
  }
}

function graphFromDetail(detail: JavaGraphDetail): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodes = detail.graph?.nodes ?? [];
  const edges = detail.graph?.edges ?? [];
  return {
    nodes: nodes.map((node, index) => {
      const id = stringValue(node.node_id ?? node.id, `node-${index + 1}`);
      return {
        id,
        name: stringValue(node.name ?? node.label ?? node.node_id ?? node.id, id),
        type: safeNodeType(node.node_type ?? node.type),
        value: numericValue(node.value ?? node.weight, 1)
      };
    }),
    edges: edges.map((edge, index) => ({
      source: stringValue(edge.source_id ?? edge.source, `source-${index + 1}`),
      target: stringValue(edge.target_id ?? edge.target, `target-${index + 1}`),
      relation: stringValue(edge.relation ?? edge.type, "关联"),
      weight: numericValue(edge.weight, 1)
    }))
  };
}

function averageQuality(items: QualityReport[]): number {
  if (!items.length) {
    return 0;
  }
  return Math.round(items.reduce((total, item) => total + item.completeness + item.validity, 0) / (items.length * 2) * 10) / 10;
}

function percentage(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 1_000) / 10;
}

function safeStore(value: string): DataAsset["store"] {
  return value === "raw" || value === "processed" || value === "features" || value === "graphs" ? value : "processed";
}

function safeNodeType(value: unknown): GraphNode["type"] {
  return value === "seat" || value === "stock" || value === "industry" || value === "concept" || value === "event" ? value : "event";
}

function safeTaskStatus(value: string): SyncTask["status"] {
  return value === "queued" || value === "running" || value === "success" || value === "failed" ? value : "failed";
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function numericValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
