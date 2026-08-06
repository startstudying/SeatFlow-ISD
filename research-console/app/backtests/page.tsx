"use client";

import { useEffect, useState } from "react";
import api, { type APIError, type Backtest } from "@/lib/api/client";
import { asAPIError, ResearchErrorPanel } from "@/lib/api/research-error";

export default function BacktestsPage() {
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [error, setError] = useState<APIError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listBacktests(1, 50)
      .then((res) => {
        setBacktests(res.items);
        setError(null);
      })
      .catch((caught: unknown) => setError(asAPIError(caught)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading...</div>;

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">Backtests</h1>
      <div className="mb-4 rounded-md bg-blue-50 p-3 text-sm text-blue-700">
        All backtest results include transaction costs. Research / backtest only.
      </div>
      {error ? (
        <ResearchErrorPanel error={error} />
      ) : backtests.length === 0 ? (
        <p role="status" className="rounded-md bg-gray-50 p-4 text-sm text-gray-600">
          No verified backtests returned.
        </p>
      ) : (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Period
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Total Return
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Sharpe
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Max DD
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Hit Rate
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Type
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {backtests.map((b) => (
              <tr key={b.backtestId} className="hover:bg-gray-50">
                <td className="whitespace-nowrap px-6 py-4 text-sm font-medium text-gray-900">
                  {b.modelId.slice(0, 8)}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {new Date(b.startDate).toLocaleDateString()} –{" "}
                  {new Date(b.endDate).toLocaleDateString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-right text-sm">
                  <span className={b.totalReturn >= 0 ? "text-green-600" : "text-red-600"}>
                    {b.totalReturn.toFixed(2)}%
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-right text-sm text-gray-900">
                  {b.sharpe.toFixed(2)}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-right text-sm text-red-600">
                  {b.maxDrawdown.toFixed(2)}%
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-right text-sm text-gray-900">
                  {(b.hitRate * 100).toFixed(1)}%
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <span className="inline-flex rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700">
                    {b.isBlind ? "Blind" : "Research"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </main>
  );
}
