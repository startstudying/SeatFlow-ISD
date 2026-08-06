import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { describe, it, expect } from "vitest";
import Home from "../app/page";

describe("Research Console smoke test", () => {
  it("renders SeatFlow V3 title after initialization", async () => {
    render(<Home />);
    expect(
      await screen.findByRole("heading", { name: /SeatFlow V3/i }),
    ).toBeInTheDocument();
  });

  it("displays the research-only truthfulness banner after initialization", async () => {
    render(<Home />);
    expect(
      await screen.findByText(/Research \/ Backtest/i),
    ).toBeInTheDocument();
  });

  it("does not display any formal prediction state after initialization", async () => {
    const { container } = render(<Home />);
    await screen.findByRole("heading", { name: /SeatFlow V3/i });
    // The root page must not render any element that claims a formal model.
    expect(container.textContent).not.toMatch(/formal prediction/i);
    expect(container.textContent).not.toMatch(/model valid/i);
  });
});
