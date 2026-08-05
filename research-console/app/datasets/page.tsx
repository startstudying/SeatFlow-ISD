"use client";

import { useEffect, useState } from "react";
import api, { type Dataset } from "@/lib/api/client";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listDatasets(1, 50)
      .then((res) => setDatasets(res.items))
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading...</div>;

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">Datasets</h1>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Version
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Rows
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Nodes
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Edges
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Sealed
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {datasets.map((d) => (
              <tr key={d.datasetId}>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">
                  {d.version}
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <span
                    className={`inline-flex rounded-full px-2 text-xs font-semibold ${
                      d.status === "sealed"
                        ? "bg-green-100 text-green-800"
                        : d.status === "draft"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                    }`}
                  >
                    {d.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {d.rowCount.toLocaleString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {d.nodeCount.toLocaleString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {d.edgeCount.toLocaleString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {d.sealedAt ? new Date(d.sealedAt).toLocaleDateString() : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
