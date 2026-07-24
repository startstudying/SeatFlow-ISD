import { describe, expect, it } from "vitest";

import { edgeWidthPercent } from "../components/GraphCanvas";

describe("graph edge width", () => {
  it("keeps large monetary weights inside the graph stage", () => {
    expect(edgeWidthPercent(12_000_000)).toBe("52%");
  });

  it("keeps normalized graph weights legible", () => {
    expect(edgeWidthPercent(0.3)).toBe("29%");
    expect(edgeWidthPercent(undefined)).toBe("29%");
  });
});
