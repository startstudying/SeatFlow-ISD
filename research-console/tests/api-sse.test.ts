import { afterEach, describe, expect, it, vi } from "vitest";

import { subscribeToTask } from "../lib/api/sse";

describe("task event API boundary", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("subscribes through the Control Center system task route", () => {
    let subscribedUrl = "";
    let subscribedWithCredentials = false;

    class FakeEventSource {
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string | URL, options?: EventSourceInit) {
        subscribedUrl = url.toString();
        subscribedWithCredentials = options?.withCredentials ?? false;
      }

      close(): void {}
    }

    vi.stubGlobal("EventSource", FakeEventSource);

    const unsubscribe = subscribeToTask("task-123", () => undefined);

    expect(subscribedUrl).toBe(
      "http://localhost:8080/api/v1/system/tasks/task-123/events",
    );
    expect(subscribedWithCredentials).toBe(true);
    unsubscribe();
  });
});
