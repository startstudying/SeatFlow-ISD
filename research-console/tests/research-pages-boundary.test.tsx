import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import BacktestsPage from "@/app/backtests/page";
import DatasetsPage from "@/app/datasets/page";
import PredictionsPage from "@/app/predictions/page";
import { APIError } from "@/lib/api/client";

const apiMocks = vi.hoisted(() => ({
  listBacktests: vi.fn(),
  listDatasets: vi.fn(),
  listPredictions: vi.fn(),
}));

vi.mock("@/lib/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api/client")>();
  return {
    ...actual,
    default: {
      ...actual.default,
      ...apiMocks,
    },
  };
});

function unavailable(operation: string): APIError {
  return new APIError(
    "RESEARCH_ARTIFACT_UNAVAILABLE",
    "Verified research artifacts are unavailable for this operation.",
    "trace-page-123",
    undefined,
    true,
    {
      status: "BLOCKED",
      operation,
      reason: "VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED",
    },
    "Configure the canonical research-engine gateway.",
    "2026-08-06T07:00:00Z",
  );
}

const pageCases = [
  {
    label: "datasets",
    Page: DatasetsPage,
    load: apiMocks.listDatasets,
    operation: "datasets.list",
    emptyMessage: "No verified datasets returned.",
  },
  {
    label: "backtests",
    Page: BacktestsPage,
    load: apiMocks.listBacktests,
    operation: "backtests.list",
    emptyMessage: "No verified backtests returned.",
  },
  {
    label: "predictions",
    Page: PredictionsPage,
    load: apiMocks.listPredictions,
    operation: "predictions.list",
    emptyMessage: "No verified predictions returned.",
  },
];

describe("research result pages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each(pageCases)(
    "renders the complete BLOCKED envelope on the $label page",
    async ({ Page, load, operation }) => {
      load.mockRejectedValue(unavailable(operation));

      render(<Page />);

      const alert = await screen.findByRole("alert");
      expect(within(alert).getByRole("heading", { name: "Research data blocked" })).toBeInTheDocument();
      expect(within(alert).getByText("RESEARCH_ARTIFACT_UNAVAILABLE")).toBeInTheDocument();
      expect(within(alert).getByText("BLOCKED")).toBeInTheDocument();
      expect(within(alert).getByText(operation)).toBeInTheDocument();
      expect(within(alert).getByText("VERIFIED_RESEARCH_GATEWAY_NOT_CONFIGURED")).toBeInTheDocument();
      expect(within(alert).getByText("Configure the canonical research-engine gateway.")).toBeInTheDocument();
      expect(within(alert).getByText("Retryable: yes")).toBeInTheDocument();
      expect(within(alert).getByText("trace-page-123")).toBeInTheDocument();
      expect(within(alert).getByText("2026-08-06T07:00:00Z")).toBeInTheDocument();
      expect(screen.queryByRole("table")).not.toBeInTheDocument();
    },
  );

  it("does not render non-string or unknown error details", async () => {
    apiMocks.listDatasets.mockRejectedValue(
      new APIError(
        "RESEARCH_ARTIFACT_UNAVAILABLE",
        "Verified research artifacts are unavailable for this operation.",
        "trace-page-123",
        undefined,
        true,
        {
          status: "BLOCKED",
          operation: "datasets.list",
          reason: { secret: "DO_NOT_RENDER_REASON" },
          internalPayload: "DO_NOT_RENDER_INTERNAL_PAYLOAD",
        },
      ),
    );

    render(<DatasetsPage />);

    const alert = await screen.findByRole("alert");
    expect(within(alert).queryByText(/DO_NOT_RENDER_REASON/)).not.toBeInTheDocument();
    expect(within(alert).queryByText(/DO_NOT_RENDER_INTERNAL_PAYLOAD/)).not.toBeInTheDocument();
  });

  it.each(pageCases)(
    "distinguishes a successful empty $label response from BLOCKED",
    async ({ Page, load, emptyMessage }) => {
      load.mockResolvedValue({
        items: [],
        totalCount: 0,
        page: 1,
        pageSize: 50,
      });

      render(<Page />);

      expect(await screen.findByText(emptyMessage)).toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    },
  );
});
