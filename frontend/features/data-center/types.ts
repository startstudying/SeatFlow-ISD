export type DataCenterSection = "overview" | "assets" | "quality" | "graph" | "versions" | "sync";

export type DataAssetStatus = "ready" | "syncing" | "degraded" | "empty";
export type SyncTaskStatus = "queued" | "running" | "success" | "failed";

export type DataAsset = {
  id: string;
  provider: string;
  dataset: string;
  store: "raw" | "processed" | "features" | "graphs";
  rows: number;
  version: string;
  owner: string;
  updatedAt: string;
  status: DataAssetStatus;
  lineageHash: string | null;
};

export type QualityReport = {
  id: string;
  provider: string;
  dataset: string;
  completeness: number;
  validity: number;
  freshnessHours: number;
  issueCount: number;
  gate: "pass" | "warn" | "block";
  checkedAt: string;
};

export type GraphNode = {
  id: string;
  name: string;
  type: "seat" | "stock" | "industry" | "concept" | "event";
  value: number;
};

export type GraphEdge = {
  source: string;
  target: string;
  relation: string;
  weight: number;
};

export type DatasetVersion = {
  id: string;
  dataset: string;
  version: string;
  tag: string;
  rows: number;
  createdAt: string;
  checksum: string;
  author: string;
};

export type SyncTask = {
  id: string;
  source: string;
  target: string;
  status: SyncTaskStatus;
  progress: number;
  startedAt: string;
  message: string;
};

export type DataCenterOverview = {
  generatedAt: string;
  provider: string;
  tradeDate: string;
  assetCount: number;
  rowCount: number;
  graphNodeCount: number;
  graphEdgeCount: number;
  qualityScore: number;
  blockedIssues: number;
  activeTasks: number;
};

export type DataCenterPayload = {
  mode: "mock" | "java-api";
  overview: DataCenterOverview;
  assets: DataAsset[];
  quality: QualityReport[];
  graph: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  versions: DatasetVersion[];
  syncTasks: SyncTask[];
};

export type AssetFilters = {
  keyword: string;
  store: "all" | DataAsset["store"];
  status: "all" | DataAssetStatus;
  page: number;
  pageSize: number;
};

export type PaginatedAssets = {
  rows: DataAsset[];
  total: number;
  page: number;
  pageSize: number;
};
