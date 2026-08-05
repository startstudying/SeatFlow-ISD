"use client";

import { useState, useEffect } from "react";
import { subscribeToAllTasks, type TaskEvent } from "@/lib/api/sse";

export default function GovernancePage() {
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const unsub = subscribeToAllTasks(
      (event) => {
        setEvents((prev) => [event, ...prev].slice(0, 100));
      },
      () => setConnected(false),
    );
    setConnected(true);
    return unsub;
  }, []);

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">Governance & Audit Trail</h1>

      <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-200 p-4">
          <div className="text-sm text-gray-500">SSE Connection</div>
          <div className="mt-1 flex items-center">
            <span
              className={`mr-2 h-2 w-2 rounded-full ${
                connected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-sm font-medium">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
        <div className="rounded-lg border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Total Events</div>
          <div className="mt-1 text-2xl font-bold">{events.length}</div>
        </div>
        <div className="rounded-lg border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Model Type</div>
          <div className="mt-1 text-sm font-medium text-blue-700">
            Research / Backtest Only
          </div>
        </div>
      </div>

      <section className="rounded-lg border border-gray-200">
        <div className="border-b border-gray-200 px-6 py-4">
          <h2 className="text-lg font-semibold">Audit Event Log</h2>
        </div>
        <div className="max-h-96 overflow-y-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                  Event
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                  Task ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                  Trace ID
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {events.map((e, i) => (
                <tr key={i}>
                  <td className="whitespace-nowrap px-6 py-3 text-xs text-gray-500">
                    {new Date(e.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="whitespace-nowrap px-6 py-3">
                    <span
                      className={`inline-flex rounded px-2 py-0.5 text-xs font-medium ${
                        e.eventType.includes("COMPLETED")
                          ? "bg-green-50 text-green-700"
                          : e.eventType.includes("FAILED")
                            ? "bg-red-50 text-red-700"
                            : e.eventType.includes("STARTED")
                              ? "bg-blue-50 text-blue-700"
                              : "bg-gray-50 text-gray-700"
                      }`}
                    >
                      {e.eventType}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-6 py-3 text-xs text-gray-500">
                    {e.taskId.slice(0, 8)}
                  </td>
                  <td className="whitespace-nowrap px-6 py-3 text-xs text-gray-400">
                    {e.traceId.slice(0, 8)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {events.length === 0 && (
            <div className="px-6 py-8 text-center text-sm text-gray-400">
              No events yet. Waiting for SSE connection...
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
