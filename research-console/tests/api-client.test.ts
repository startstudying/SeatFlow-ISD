import { afterEach, describe, expect, it, vi } from "vitest";

import { APIError, api } from "../lib/api/client";

function respond(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

describe("Research Console API boundary", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("preserves remediation and blocked details from the SeatFlow error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        respond(503, {
          code: "RESEARCH_ARTIFACT_UNAVAILABLE",
          message: "Verified research artifact is unavailable",
          details: {
            status: "BLOCKED",
            operation: "backtests.get",
            reason: "VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED",
          },
          traceId: "trace-123",
          timestamp: "2026-08-06T07:00:00Z",
          retryable: true,
          remediation: "Configure the canonical research gateway.",
        }),
      ),
    );

    const error = await api.getBacktest("bt-missing").catch((caught) => caught);

    expect(error).toBeInstanceOf(APIError);
    expect(error).toMatchObject({
      code: "RESEARCH_ARTIFACT_UNAVAILABLE",
      traceId: "trace-123",
      timestamp: "2026-08-06T07:00:00Z",
      retryable: true,
      remediation: "Configure the canonical research gateway.",
      details: {
        status: "BLOCKED",
        operation: "backtests.get",
        reason: "VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED",
      },
    });
  });

  it("returns the control-plane BLOCKED task state without replacing its error fields", async () => {
    const blockedState = {
      taskId: "task-123",
      status: "BLOCKED",
      progress: 0,
      updatedAt: "2026-08-06T07:00:00Z",
      result: null,
      error: {
        code: "RESEARCH_ARTIFACT_UNAVAILABLE",
        message: "Dataset artifact is unavailable",
        traceId: "trace-task-123",
        timestamp: "2026-08-06T07:00:00Z",
        details: { datasetId: "dataset-missing" },
        retryable: true,
        remediation: "Build and seal the dataset first.",
      },
      blockedBy: ["DATASET_NOT_READY"],
    };
    const fetchTask = vi.fn().mockResolvedValue(respond(200, blockedState));
    vi.stubGlobal("fetch", fetchTask);

    const state = await api.getTask("task-123");

    expect(state).toEqual(blockedState);
    expect(fetchTask).toHaveBeenCalledWith(
      "http://localhost:8080/api/v1/system/tasks/task-123",
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
      },
    );
  });
});
