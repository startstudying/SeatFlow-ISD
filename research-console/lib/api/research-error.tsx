import { APIError } from "./client";

type SafeDetailKey = "status" | "operation" | "reason";

function safeString(value: unknown): string {
  return typeof value === "string" && value.length > 0 ? value : "not provided";
}

function safeDetail(error: APIError, key: SafeDetailKey): string {
  const value = error.details?.[key];
  return safeString(value);
}

export function asAPIError(caught: unknown): APIError {
  if (caught instanceof APIError) return caught;
  return new APIError(
    "UNEXPECTED_CLIENT_ERROR",
    caught instanceof Error ? caught.message : "Unexpected API client error",
    "",
  );
}

export function ResearchErrorPanel({ error }: { error: APIError }) {
  return (
    <section
      role="alert"
      className="rounded-md border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950"
    >
      <h2 className="mb-3 text-lg font-semibold">Research data blocked</h2>
      <p className="mb-3">{error.message}</p>
      <dl className="grid gap-2 sm:grid-cols-[9rem_1fr]">
        <dt className="font-medium">Code</dt>
        <dd className="font-mono">{safeString(error.code)}</dd>
        <dt className="font-medium">Status</dt>
        <dd>{safeDetail(error, "status")}</dd>
        <dt className="font-medium">Operation</dt>
        <dd className="font-mono">{safeDetail(error, "operation")}</dd>
        <dt className="font-medium">Reason</dt>
        <dd className="font-mono">{safeDetail(error, "reason")}</dd>
        <dt className="font-medium">Remediation</dt>
        <dd>{safeString(error.remediation)}</dd>
        <dt className="font-medium">Retryable</dt>
        <dd>{`Retryable: ${error.retryable ? "yes" : "no"}`}</dd>
        <dt className="font-medium">Trace ID</dt>
        <dd className="font-mono">{safeString(error.traceId)}</dd>
        <dt className="font-medium">Timestamp</dt>
        <dd>{safeString(error.timestamp)}</dd>
      </dl>
    </section>
  );
}
