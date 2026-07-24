"use client";

export default function ErrorPage({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <section className="card state-card">
      <div>
        <h3>前端渲染失败</h3>
        <p className="muted">{error.message}</p>
        <button className="pill" onClick={reset} type="button">
          重试
        </button>
      </div>
    </section>
  );
}
