"use client";

import { useEffect, useState } from "react";
import api, { type APIError, type Prediction } from "@/lib/api/client";
import { asAPIError, ResearchErrorPanel } from "@/lib/api/research-error";
import { formatPredictionLabel } from "@/lib/config/truthfulness";

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState<APIError | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listPredictions(1, 50)
      .then((res) => {
        setPredictions(res.items);
        setError(null);
      })
      .catch((caught: unknown) => setError(asAPIError(caught)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading...</div>;

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">Predictions</h1>
      <div className="mb-4 rounded-md bg-blue-50 p-3 text-sm text-blue-700">
        All predictions are research / backtest only. No formal predictions.
      </div>
      {error ? (
        <ResearchErrorPanel error={error} />
      ) : predictions.length === 0 ? (
        <p role="status" className="rounded-md bg-gray-50 p-4 text-sm text-gray-600">
          No verified predictions returned.
        </p>
      ) : (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Stock
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Label
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                R1
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                R3
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                R5
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Tradeability
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {predictions.map((p) => (
              <tr key={p.predictionId} className="hover:bg-gray-50">
                <td className="whitespace-nowrap px-6 py-4 text-sm font-medium text-gray-900">
                  {p.stockId}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {new Date(p.predictionDate).toLocaleDateString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <span className="inline-flex rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700">
                    {formatPredictionLabel(p)}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">
                  {p.cumulativeReturn1.toFixed(2)}%
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">
                  {p.cumulativeReturn3.toFixed(2)}%
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">
                  {p.cumulativeReturn5.toFixed(2)}%
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {p.tradeability}
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
