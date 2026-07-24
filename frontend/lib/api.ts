export type DataScope = "real" | "sample" | "synthetic" | "none" | "unknown";
export type SeatDetailScope = "complete_seat" | "institutional_only" | "synthetic" | "none" | "unknown";
export type ModelValidityStatus = "valid" | "invalid" | "not_available" | "degraded" | "smoke_only" | "unknown";

export type EvidenceMeta = {
  data_scope: DataScope;
  seat_detail_scope: SeatDetailScope;
  lineage_hash: string | null;
  risk_flags: string[];
  model_validity_status: ModelValidityStatus;
  source: "api" | "stub";
};

export type UiState<T> =
  | { kind: "loading"; meta: EvidenceMeta }
  | { kind: "empty"; title: string; message: string; meta: EvidenceMeta }
  | { kind: "error"; title: string; message: string; meta: EvidenceMeta }
  | { kind: "degraded"; reason: string; data: T; meta: EvidenceMeta }
  | { kind: "real"; data: T; meta: EvidenceMeta };

export type SnapshotSummary = {
  provider: string;
  snapshot_count: number;
  datasets: Array<Record<string, unknown>>;
  data_scope?: DataScope;
  lineage_hash?: string | null;
  risk_flags?: string[];
  model_validity_status?: ModelValidityStatus;
};

export type GraphSnapshot = {
  manifest?: {
    row_count?: number;
    content_sha256?: string;
    sha256?: string;
    metadata?: Record<string, unknown>;
  };
  graph?: {
    as_of_date?: string;
    nodes?: Array<{ node_type?: string; node_id?: string; features?: Record<string, unknown> }>;
    edges?: Array<{
      source_type?: string;
      source_id?: string;
      target_type?: string;
      target_id?: string;
      relation?: string;
      weight?: number;
    }>;
  };
  data_scope?: DataScope;
  lineage_hash?: string | null;
  risk_flags?: string[];
  model_validity_status?: ModelValidityStatus;
};

export type MarketPayload = {
  summary: SnapshotSummary;
  graph: GraphSnapshot | null;
};

export type ApiProblem = {
  status: number;
  message: string;
  meta?: EvidenceMeta;
};

const unknownMeta: EvidenceMeta = {
  data_scope: "unknown",
  seat_detail_scope: "unknown",
  lineage_hash: null,
  risk_flags: ["api_contract_not_verified"],
  model_validity_status: "unknown",
  source: "api"
};

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_SEATFLOW_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8080";
}

export function defaultProvider(): string {
  return process.env.NEXT_PUBLIC_SEATFLOW_PROVIDER ?? "sample";
}

export function defaultTradeDate(): string {
  return process.env.NEXT_PUBLIC_SEATFLOW_TRADE_DATE ?? "2026-01-15";
}

export function defaultModelId(): string | null {
  return nonEmptyEnv("NEXT_PUBLIC_SEATFLOW_DEFAULT_MODEL_ID");
}

export function defaultRunId(): string | null {
  return nonEmptyEnv("NEXT_PUBLIC_SEATFLOW_DEFAULT_RUN_ID");
}

export function loadingState<T>(): UiState<T> {
  return { kind: "loading", meta: unknownMeta };
}

export function metadataFromPayload(payload: unknown, fallback: Partial<EvidenceMeta> = {}): EvidenceMeta {
  const record = asRecord(payload);
  const manifest = asRecord(record.manifest);
  const metadata = asRecord(manifest.metadata);
  const dataScope = coerceScope(record.data_scope ?? metadata.data_scope ?? fallback.data_scope);
  const seatDetailScope = coerceSeatDetailScope(
    record.seat_detail_scope ??
      record.seatDetailScope ??
      metadata.seat_detail_scope ??
      metadata.seatDetailScope ??
      fallback.seat_detail_scope
  );
  const riskFlags = coerceStringArray(record.risk_flags ?? metadata.risk_flags ?? fallback.risk_flags);
  const lineageHash =
    stringOrNull(record.lineage_hash ?? record.lineageHash) ??
    stringOrNull(metadata.lineage_hash ?? metadata.lineageHash) ??
    stringOrNull(manifest.content_sha256) ??
    stringOrNull(manifest.sha256) ??
    fallback.lineage_hash ??
    null;
  const modelStatus = coerceModelStatus(
    record.model_validity_status ??
      record.modelValidityStatus ??
      metadata.model_validity_status ??
      metadata.modelValidityStatus ??
      fallback.model_validity_status
  );

  return {
    data_scope: dataScope,
    seat_detail_scope: seatDetailScope,
    lineage_hash: lineageHash,
    risk_flags: riskFlags.length ? riskFlags : dataScope === "real" ? [] : [`data_scope_${dataScope}`],
    model_validity_status: modelStatus,
    source: fallback.source ?? "api"
  };
}

export function classifyPayload<T>(payload: T, meta: EvidenceMeta): UiState<T> {
  if (
    meta.source === "api" &&
    meta.data_scope === "real" &&
    meta.seat_detail_scope === "complete_seat" &&
    meta.model_validity_status === "valid" &&
    meta.lineage_hash !== null &&
    meta.risk_flags.length === 0
  ) {
    return { kind: "real", data: payload, meta };
  }

  return {
    kind: "degraded",
    reason: "当前响应未同时证明真实数据、完整席位、有效模型、血缘和无风险标记，不能展示为真实预测结论。",
    data: payload,
    meta
  };
}

export async function fetchMarketState(): Promise<UiState<MarketPayload>> {
  const provider = defaultProvider();
  const tradeDate = defaultTradeDate();
  const summaryResult = await getJson<SnapshotSummary>(`/api/v1/providers/${provider}/snapshot-summary`);

  if (!summaryResult.ok) {
    return problemToState(summaryResult.problem, "市场总览暂无可展示数据");
  }

  const graphResult = await getJson<GraphSnapshot>(`/api/v1/providers/${provider}/graph/${tradeDate}`);
  const graph = graphResult.ok ? graphResult.value : null;
  const payload = { summary: summaryResult.value, graph };
  const meta = metadataFromPayload(graph ?? summaryResult.value, {
    data_scope: provider === "sample" ? "sample" : "unknown",
    seat_detail_scope: provider === "sample" ? "synthetic" : "unknown",
    model_validity_status: provider === "sample" ? "smoke_only" : "unknown",
    risk_flags: graphResult.ok ? [] : [`graph_${tradeDate}_not_available`]
  });
  return classifyPayload(payload, meta);
}

export async function fetchCapitalState(): Promise<UiState<GraphSnapshot>> {
  const provider = defaultProvider();
  const tradeDate = defaultTradeDate();
  const result = await getJson<GraphSnapshot>(`/api/v1/providers/${provider}/graph/${tradeDate}`);

  if (!result.ok) {
    return problemToState(result.problem, "资金传播图暂无图快照");
  }

  return classifyPayload(
    result.value,
    metadataFromPayload(result.value, {
      data_scope: provider === "sample" ? "sample" : "unknown",
      seat_detail_scope: provider === "sample" ? "synthetic" : "unknown",
      model_validity_status: provider === "sample" ? "smoke_only" : "unknown"
    })
  );
}

export async function fetchPredictionState(): Promise<UiState<unknown>> {
  const modelId = defaultModelId();
  if (modelId === null) {
    return configuredEmptyState("股票预测分析等待 NEXT_PUBLIC_SEATFLOW_DEFAULT_MODEL_ID。", "model_id_not_configured");
  }
  const result = await getJson<unknown>(`/api/v1/models/${encodeURIComponent(modelId)}`);

  if (!result.ok) {
    return problemToState(result.problem, "股票预测分析等待模型服务契约");
  }

  return classifyPayload(result.value, metadataFromPayload(result.value));
}

export async function fetchExperimentState(): Promise<UiState<unknown>> {
  const runId = defaultRunId();
  if (runId === null) {
    return configuredEmptyState("实验中心等待 NEXT_PUBLIC_SEATFLOW_DEFAULT_RUN_ID。", "run_id_not_configured");
  }
  const result = await getJson<unknown>(`/api/v1/runs/${encodeURIComponent(runId)}`);

  if (!result.ok) {
    return problemToState(result.problem, "实验中心等待实验 lineage/run 契约");
  }

  return classifyPayload(result.value, metadataFromPayload(result.value));
}

export function buildStubState<T>(kind: UiState<T>["kind"], data: T, overrides: Partial<EvidenceMeta> = {}): UiState<T> {
  const meta: EvidenceMeta = {
    data_scope: "synthetic",
    seat_detail_scope: "synthetic",
    lineage_hash: "local-ui-stub",
    risk_flags: ["explicit_ui_stub_not_real_data"],
    model_validity_status: "smoke_only",
    source: "stub",
    ...overrides
  };

  if (kind === "loading") {
    return { kind: "loading", meta };
  }
  if (kind === "empty") {
    return { kind: "empty", title: "本地空态 Stub", message: "仅用于验证 UI 空态，不代表真实数据。", meta };
  }
  if (kind === "error") {
    return { kind: "error", title: "本地错误态 Stub", message: "仅用于验证 UI 错误态，不代表真实 API。", meta };
  }
  if (kind === "real") {
    return {
      kind: "degraded",
      reason: "显式 UI Stub 永远不能表示真实有效模型。",
      data,
      meta
    };
  }
  return { kind: "degraded", reason: "本地显式 Stub 降级态。", data, meta };
}

async function getJson<T>(path: string): Promise<{ ok: true; value: T } | { ok: false; problem: ApiProblem }> {
  const url = `${apiBaseUrl()}${path}`;
  try {
    const response = await fetch(url, { cache: "no-store", headers: { accept: "application/json" } });
    const payload: unknown = await response.json().catch(() => null);
    const value = unwrapApiEnvelope(payload);
    if (!response.ok) {
      const record = asRecord(value);
      return {
        ok: false,
        problem: {
          status: response.status,
          message: errorMessage(record, response.statusText || "API request failed"),
          meta: metadataFromPayload(value)
        }
      };
    }
    return { ok: true, value: value as T };
  } catch (error) {
    return {
      ok: false,
      problem: { status: 0, message: error instanceof Error ? error.message : "Network request failed" }
    };
  }
}

function problemToState<T>(problem: ApiProblem, title: string): UiState<T> {
  if (problem.status === 404 || problem.status === 409) {
    return {
      kind: "empty",
      title,
      message: `${problem.status} ${problem.message}`.trim(),
      meta: {
        ...(problem.meta ?? unknownMeta),
        source: "api",
        data_scope: problem.meta?.data_scope ?? "none",
        seat_detail_scope: problem.meta?.seat_detail_scope ?? "none",
        model_validity_status: problem.meta?.model_validity_status ?? "not_available",
        risk_flags: problem.meta?.risk_flags.length ? problem.meta.risk_flags : ["api_returned_no_data"]
      }
    };
  }

  return {
    kind: "error",
    title: "API 请求失败",
    message: problem.status ? `${problem.status} ${problem.message}` : problem.message,
    meta: { ...unknownMeta, risk_flags: ["api_request_failed"] }
  };
}

function configuredEmptyState<T>(message: string, riskFlag: string): UiState<T> {
  return {
    kind: "empty",
    title: "运行配置尚未就绪",
    message,
    meta: {
      ...unknownMeta,
      data_scope: "none",
      seat_detail_scope: "none",
      model_validity_status: "not_available",
      risk_flags: [riskFlag]
    }
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

export function unwrapApiEnvelope(payload: unknown): unknown {
  const envelope = asRecord(payload);
  return "data" in envelope ? envelope.data : payload;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function coerceStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function coerceScope(value: unknown): DataScope {
  return value === "real" ||
    value === "synthetic" ||
    value === "sample" ||
    value === "none"
    ? value
    : "unknown";
}

function coerceSeatDetailScope(value: unknown): SeatDetailScope {
  return value === "complete_seat" ||
    value === "institutional_only" ||
    value === "synthetic" ||
    value === "none"
    ? value
    : "unknown";
}

function errorMessage(record: Record<string, unknown>, fallback: string): string {
  const value = record.error ?? record.detail ?? record.message;
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function nonEmptyEnv(name: string): string | null {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

function coerceModelStatus(value: unknown): ModelValidityStatus {
  return value === "valid" ||
    value === "invalid" ||
    value === "not_available" ||
    value === "degraded" ||
    value === "smoke_only"
    ? value
    : "unknown";
}
