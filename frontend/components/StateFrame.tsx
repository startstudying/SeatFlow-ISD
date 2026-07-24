import type { ReactNode } from "react";
import type { UiState } from "@/lib/api";
import { EvidenceStrip } from "./EvidenceStrip";

export function StateFrame<T>({
  state,
  children
}: {
  state: UiState<T>;
  children: (data: T, degradedReason?: string) => ReactNode;
}) {
  if (state.kind === "loading") {
    return (
      <>
        <EvidenceStrip meta={state.meta} />
        <section className="card state-card" aria-live="polite">
          <div className="skeleton">
            <h3>正在读取接口证据</h3>
            <p className="muted">页面不会在证据未返回前填充推测结果。</p>
            <div />
            <div />
            <div />
          </div>
        </section>
      </>
    );
  }

  if (state.kind === "empty" || state.kind === "error") {
    return (
      <>
        <EvidenceStrip meta={state.meta} />
        <section className="card state-card">
          <div>
            <h3>{state.title}</h3>
            <p className="muted">{state.message}</p>
            <p className="fine">未返回可验证数据时不展示模拟指标，也不生成预测或交易结论。</p>
          </div>
        </section>
      </>
    );
  }

  return (
    <>
      <EvidenceStrip meta={state.meta} />
      {children(state.data, state.kind === "degraded" ? state.reason : undefined)}
    </>
  );
}
