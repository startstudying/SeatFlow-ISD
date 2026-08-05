"use client";

import { useEffect, useState } from "react";
import api, { type Experiment } from "@/lib/api/client";
import Link from "next/link";

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listExperiments(1, 50)
      .then((res) => setExperiments(res.items))
      .catch(() => setExperiments([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8">Loading...</div>;

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Experiments</h1>
        <button
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          onClick={() => {
            /* Open create experiment modal */
          }}
        >
          New Experiment
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Name
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Dataset
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Created
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Completed
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {experiments.map((e) => (
              <tr key={e.experimentId} className="hover:bg-gray-50">
                <td className="whitespace-nowrap px-6 py-4 text-sm">
                  <Link
                    href={`/experiments/${e.experimentId}`}
                    className="text-blue-600 hover:underline"
                  >
                    {e.name}
                  </Link>
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <span
                    className={`inline-flex rounded-full px-2 text-xs font-semibold ${
                      e.status === "completed"
                        ? "bg-green-100 text-green-800"
                        : e.status === "running"
                          ? "bg-blue-100 text-blue-800"
                          : e.status === "failed"
                            ? "bg-red-100 text-red-800"
                            : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {e.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {e.datasetId.slice(0, 8)}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {new Date(e.createdAt).toLocaleString()}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">
                  {e.completedAt
                    ? new Date(e.completedAt).toLocaleString()
                    : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
