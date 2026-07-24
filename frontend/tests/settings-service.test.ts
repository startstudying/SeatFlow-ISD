import { describe, expect, it } from "vitest";
import { fetchMockSettings } from "../features/settings/mock-service";

describe("settings mock adapter", () => {
  it("keeps data-source secrets masked while exposing operational metadata", async () => {
    const payload = await fetchMockSettings();
    const source = payload.dataSources.find((item) => item.id === "tushare");

    expect(payload.mode).toBe("mock");
    expect(source?.apiKeyMasked).toContain("仅显示状态");
    expect(source?.apiKeyMasked).not.toMatch(/[a-f0-9]{32,}/i);
    expect(source?.endpoint).toContain("https://");
  });

  it("ships all required settings views through one typed payload", async () => {
    const payload = await fetchMockSettings();

    expect(payload.settings).not.toHaveLength(0);
    expect(payload.models).not.toHaveLength(0);
    expect(payload.experiments).not.toHaveLength(0);
    expect(payload.services).not.toHaveLength(0);
    expect(payload.users).not.toHaveLength(0);
    expect(payload.auditLogs).not.toHaveLength(0);
  });
});
