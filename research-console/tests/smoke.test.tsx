import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { describe, it, expect } from "vitest";
import Home from "../app/page";

describe("Research Console smoke test", () => {
  it("renders SeatFlow V3 title", () => {
    render(<Home />);
    expect(
      screen.getByRole("heading", { name: /SeatFlow V3/i }),
    ).toBeInTheDocument();
  });

  it("displays truthfulness banner stating no formal model is active", () => {
    render(<Home />);
    expect(
      screen.getByText(/No formal model is active/i),
    ).toBeInTheDocument();
  });

  it("does not display any formal prediction state", () => {
    const { container } = render(<Home />);
    // The root page must not render any element that claims a formal model.
    expect(container.textContent).not.toMatch(/formal prediction/i);
    expect(container.textContent).not.toMatch(/model valid/i);
  });
});
