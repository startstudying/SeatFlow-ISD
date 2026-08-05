"use client";

import { useEffect, useState } from "react";
import api, { type Model } from "@/lib/api/client";

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listModels(1, 50)
      .then((res) => setModels(res.items))
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading...</div>;

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Models</h1>
        <div className="inline-flex items-center rounded-md bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
          Research / Backtest Only
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Variant
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Version
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                model_valid
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Promotion
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Paper Trading
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Gates
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {models.map((m) => (
              <tr key={m.modelId} className="hover:bg-gray-50">
                <td className="whitespace-nowrap px-6 py-4 text-sm font-medium text-gray-900">
                  {m.variant}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {m.version}
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <span
                    className={`inline-flex rounded-full px-2 text-xs font-semibold ${
                      m.status === "promoted"
                        ? "bg-green-100 text-green-800"
                        : m.status === "candidate"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {m.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <span
                    className={
                      m.modelValid ? "text-green-600" : "text-red-600"
                    }
                  >
                    {m.modelValid ? "✓" : "✗"}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <span
                    className={
                      m.promotionAllowed
                        ? "text-green-600"
                        : "text-gray-400"
                    }
                  >
                    {m.promotionAllowed ? "Allowed" : "Blocked"}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <span
                    className={
                      m.paperTradingAllowed
                        ? "text-green-600"
                        : "text-gray-400"
                    }
                  >
                    {m.paperTradingAllowed ? "Allowed" : "Blocked"}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(m.gateResults).map(([gate, passed]) => (
                      <span
                        key={gate}
                        className={`inline-flex rounded px-1.5 py-0.5 text-xs ${
                          passed
                            ? "bg-green-50 text-green-700"
                            : "bg-red-50 text-red-700"
                        }`}
                        title={gate}
                      >
                        {passed ? "✓" : "✗"} {gate}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
