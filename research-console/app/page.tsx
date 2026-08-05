/**
 * SeatFlow V3 Research Console — root page.
 *
 * Displays the application name and a truthfulness banner stating that no
 * formal model is active. This is required by the V3 architecture: the UI must
 * never present synthetic or placeholder results as formal predictions.
 */
export default function HomePage() {
  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <h1 className="text-4xl font-bold tracking-tight">SeatFlow V3</h1>
      <p className="mt-4 text-lg text-gray-600">
        Seat-Stock Dynamic Heterogeneous Graph Network for Dragon-Tiger-List
        capital propagation and short-term return prediction.
      </p>
      <div
        className="mt-8 rounded-lg border border-amber-300 bg-amber-50 px-6 py-4"
        role="status"
        aria-live="polite"
      >
        <p className="text-sm font-medium text-amber-800">
          No formal model is active
        </p>
        <p className="mt-1 text-sm text-amber-700">
          All research results shown in this console are clearly labeled as
          non-formal until a model passes the full validity gate and is
          promoted through the Control Center.
        </p>
      </div>
    </main>
  );
}
