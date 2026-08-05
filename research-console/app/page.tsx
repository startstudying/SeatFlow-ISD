"use client";

import { useEffect, useState } from "react";
import api, { type Experiment, type Project, type Model } from "@/lib/api/client";
import { getModelTypeLabel, formatPredictionLabel } from "@/lib/config/truthfulness";
import Link from "next/link";

export default function HomePage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [projRes, expRes, modRes] = await Promise.all([
          api.listProjects(1, 5).catch(() => ({ items: [] })),
          api.listExperiments(1, 5).catch(() => ({ items: [] })),
          api.listModels(1, 5).catch(() => ({ items: [] })),
        ]);
        setProjects(projRes.items);
        setExperiments(expRes.items);
        setModels(modRes.items);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          SeatFlow V3 Research Console
        </h1>
        <p className="mt-2 text-sm text-gray-600">
          Dragon-Tiger-List Capital Propagation Research Platform
        </p>
        <div className="mt-2 inline-flex items-center rounded-md bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
          {getModelTypeLabel()}
        </div>
      </header>

      {error && (
        <div className="mb-4 rounded-md bg-red-50 p-4">
          <p className="text-sm text-red-700">Error: {error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <section className="rounded-lg border border-gray-200 p-6">
          <h2 className="mb-4 text-lg font-semibold text-gray-900">
            Projects ({projects.length})
          </h2>
          <ul className="space-y-2">
            {projects.map((p) => (
              <li key={p.projectId}>
                <Link
                  href={`/projects/${p.projectId}`}
                  className="text-blue-600 hover:underline"
                >
                  {p.name}
                </Link>
              </li>
            ))}
          </ul>
          <Link
            href="/projects"
            className="mt-4 inline-block text-sm text-blue-600 hover:underline"
          >
            View all →
          </Link>
        </section>

        <section className="rounded-lg border border-gray-200 p-6">
          <h2 className="mb-4 text-lg font-semibold text-gray-900">
            Recent Experiments ({experiments.length})
          </h2>
          <ul className="space-y-2">
            {experiments.map((e) => (
              <li key={e.experimentId} className="text-sm">
                <span
                  className={`inline-flex h-2 w-2 rounded-full mr-2 ${
                    e.status === "completed"
                      ? "bg-green-500"
                      : e.status === "running"
                        ? "bg-blue-500"
                        : e.status === "failed"
                          ? "bg-red-500"
                          : "bg-gray-400"
                  }`}
                />
                {e.name}
              </li>
            ))}
          </ul>
          <Link
            href="/experiments"
            className="mt-4 inline-block text-sm text-blue-600 hover:underline"
          >
            View all →
          </Link>
        </section>

        <section className="rounded-lg border border-gray-200 p-6">
          <h2 className="mb-4 text-lg font-semibold text-gray-900">
            Models ({models.length})
          </h2>
          <ul className="space-y-2">
            {models.map((m) => (
              <li key={m.modelId} className="text-sm">
                <span className="font-medium">{m.variant}</span>
                <span className="ml-2 text-xs text-gray-500">{m.version}</span>
                {m.status === "promoted" && (
                  <span className="ml-2 inline-flex items-center rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-700">
                    Promoted
                  </span>
                )}
              </li>
            ))}
          </ul>
          <Link
            href="/models"
            className="mt-4 inline-block text-sm text-blue-600 hover:underline"
          >
            View all →
          </Link>
        </section>
      </div>
    </main>
  );
}
