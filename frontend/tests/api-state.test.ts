import { describe, expect, it } from "vitest";
import {
  buildStubState,
  classifyPayload,
  metadataFromPayload,
  unwrapApiEnvelope
} from "../lib/api";

describe("metadata and UI state classification", () => {
  it("only marks complete-seat API evidence with valid model and lineage as real", () => {
    const meta = metadataFromPayload({
      data_scope: "real",
      seat_detail_scope: "complete_seat",
      lineage_hash: "abc123",
      risk_flags: [],
      model_validity_status: "valid"
    });

    const state = classifyPayload({ ok: true }, meta);

    expect(state.kind).toBe("real");
    expect(state.meta.lineage_hash).toBe("abc123");
  });

  it("keeps sample or smoke payloads degraded instead of real", () => {
    const meta = metadataFromPayload({
      data_scope: "sample",
      seat_detail_scope: "synthetic",
      lineage_hash: "sample-hash",
      risk_flags: ["smoke_baseline_not_for_trading"],
      model_validity_status: "smoke_only"
    });

    const state = classifyPayload({ ok: true }, meta);

    expect(state.kind).toBe("degraded");
    expect(state.meta.risk_flags).toContain("smoke_baseline_not_for_trading");
  });

  it("requires explicit stub source for local UI state fixtures", () => {
    const state = buildStubState("degraded", { panel: "preview" });

    expect(state.kind).toBe("degraded");
    expect(state.meta.source).toBe("stub");
    expect(state.meta.risk_flags).toContain("explicit_ui_stub_not_real_data");
  });

  it("never allows a local stub to enter the real state", () => {
    const state = buildStubState("real", { panel: "preview" });

    expect(state.kind).toBe("degraded");
    expect(state.meta.data_scope).toBe("synthetic");
  });

  it("unwraps the Java API envelope before classifying data", () => {
    const payload = unwrapApiEnvelope({ data: { data_scope: "institutional_only" } });

    expect(payload).toEqual({ data_scope: "institutional_only" });
  });
});
