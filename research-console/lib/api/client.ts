/**
 * API client for SeatFlow V3 Research Console.
 *
 * Communicates with the Java Control Center. Never reads secrets or
 * direct MinIO/PostgreSQL. All data flows through Java.
 */

export interface ErrorResponse {
  code: string;
  message: string;
  traceId: string;
  taskId?: string;
  details?: Record<string, unknown>;
  retryable: boolean;
  remediation?: string;
  timestamp?: string;
}

export type TaskStatus =
  | "REQUESTED"
  | "QUEUED"
  | "RUNNING"
  | "COMPLETED"
  | "CANCELLED"
  | "FAILED"
  | "BLOCKED";

export interface TaskState {
  taskId: string;
  status: TaskStatus;
  progress?: number;
  updatedAt: string;
  result?: Record<string, unknown> | null;
  error?: ErrorResponse | null;
  blockedBy?: string[];
}

export interface PaginatedResponse<T> {
  items: T[];
  totalCount: number;
  page: number;
  pageSize: number;
}

export interface Project {
  projectId: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
  owner: string;
}

export interface Dataset {
  datasetId: string;
  version: string;
  status: "draft" | "sealed" | "invalid";
  sourceManifestHash: string;
  featureSchemaHash: string;
  labelPolicyHash: string;
  rowCount: number;
  nodeCount: number;
  edgeCount: number;
  qualityGateResults: Record<string, boolean>;
  sealedAt?: string;
  createdAt: string;
}

export interface Experiment {
  experimentId: string;
  projectId: string;
  name: string;
  status: "running" | "completed" | "failed" | "blocked";
  datasetId: string;
  configHash: string;
  createdAt: string;
  completedAt?: string;
}

export interface Model {
  modelId: string;
  version: string;
  variant: string;
  status: "candidate" | "promoted" | "retired";
  promotionAllowed: boolean;
  paperTradingAllowed: boolean;
  modelValid: boolean;
  gateResults: Record<string, boolean>;
  createdAt: string;
  promotedAt?: string;
}

export interface Prediction {
  predictionId: string;
  stockId: string;
  modelId: string;
  datasetId: string;
  predictionDate: string;
  basePath: number[];
  directPath: number[];
  spilloverPath: number[];
  finalPath: number[];
  cumulativeReturn1: number;
  cumulativeReturn3: number;
  cumulativeReturn5: number;
  directionProbability: number[];
  uncertainty: number[];
  tradeability: string;
  isFormal: boolean;
  isBlind: boolean;
  costIncluded: boolean;
}

export interface Backtest {
  backtestId: string;
  modelId: string;
  datasetId: string;
  startDate: string;
  endDate: string;
  totalReturn: number;
  sharpe: number;
  maxDrawdown: number;
  costBefore: number;
  costAfter: number;
  hitRate: number;
  isFormal: boolean;
  isBlind: boolean;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080/api/v1";

export class APIError extends Error {
  constructor(
    public code: string,
    message: string,
    public traceId: string,
    public taskId?: string,
    public retryable: boolean = false,
    public details?: Record<string, unknown>,
    public remediation?: string,
    public timestamp?: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let body: ErrorResponse | null = null;
    try {
      body = (await response.json()) as ErrorResponse;
    } catch {
      // ignore parse errors
    }
    if (body) {
      throw new APIError(
        body.code,
        body.message,
        body.traceId,
        body.taskId,
        body.retryable,
        body.details,
        body.remediation,
        body.timestamp,
      );
    }
    throw new APIError(
      "HTTP_ERROR",
      `HTTP ${response.status}`,
      "",
      undefined,
      response.status >= 500,
    );
  }
  return (await response.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
  });
  return handleResponse<T>(response);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: body ? JSON.stringify(body) : undefined,
  });
  return handleResponse<T>(response);
}

export const api = {
  // Projects
  listProjects: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Project>>(`/projects?page=${page}&pageSize=${pageSize}`),
  getProject: (id: string) => get<Project>(`/projects/${id}`),

  // Datasets
  listDatasets: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Dataset>>(`/datasets?page=${page}&pageSize=${pageSize}`),
  getDataset: (id: string) => get<Dataset>(`/datasets/${id}`),

  // Experiments
  listExperiments: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Experiment>>(`/experiments?page=${page}&pageSize=${pageSize}`),
  createExperiment: (body: Partial<Experiment>) =>
    post<Experiment>("/experiments", body),
  getExperiment: (id: string) => get<Experiment>(`/experiments/${id}`),

  // Models
  listModels: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Model>>(`/models?page=${page}&pageSize=${pageSize}`),
  getModel: (id: string) => get<Model>(`/models/${id}`),

  // Predictions
  listPredictions: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Prediction>>(`/predictions?page=${page}&pageSize=${pageSize}`),
  getPrediction: (id: string) => get<Prediction>(`/predictions/${id}`),

  // Backtests
  listBacktests: (page = 1, pageSize = 20) =>
    get<PaginatedResponse<Backtest>>(`/backtests?page=${page}&pageSize=${pageSize}`),
  getBacktest: (id: string) => get<Backtest>(`/backtests/${id}`),

  // Control-plane task state. BLOCKED/error fields are returned unchanged.
  getTask: (id: string) => get<TaskState>(`/system/tasks/${id}`),
};

export default api;
