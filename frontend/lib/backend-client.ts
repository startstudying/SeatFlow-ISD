export type ApiEnvelope<T> = {
  code: string;
  message: string;
  data: T;
  traceId: string;
  timestamp: string;
};

export type PageResult<T> = {
  items: T[];
  page: number;
  size: number;
  total: number;
};

export class BackendApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly traceId?: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "BackendApiError";
  }
}

export function backendUrl(path: string): string {
  const base = process.env.NEXT_PUBLIC_SEATFLOW_BACKEND_BASE_URL ?? "/backend";
  return `${base.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

export async function backendRequest<T>(
  path: string,
  init: globalThis.RequestInit = {}
): Promise<T> {
  const response = await fetch(backendUrl(path), {
    ...init,
    headers: {
      accept: "application/json",
      ...(init.body ? { "content-type": "application/json" } : {}),
      ...init.headers
    },
    cache: "no-store"
  });
  const envelope = (await response.json().catch(() => null)) as ApiEnvelope<T> | null;
  if (!response.ok || !envelope || envelope.code !== "OK") {
    throw new BackendApiError(
      envelope?.message ?? "业务服务请求失败",
      envelope?.code ?? "BACKEND_REQUEST_FAILED",
      envelope?.traceId,
      response.status
    );
  }
  return envelope.data;
}

export function backendGet<T>(path: string): Promise<T> {
  return backendRequest<T>(path);
}

export function backendMutation<T>(
  path: string,
  method: "POST" | "PUT" | "PATCH" | "DELETE",
  body?: unknown,
  version?: number
): Promise<T> {
  const idempotencyKey = crypto.randomUUID();
  return backendRequest<T>(path, {
    method,
    headers: {
      "idempotency-key": idempotencyKey,
      ...(version === undefined ? {} : { "if-match": String(version) })
    },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
}
