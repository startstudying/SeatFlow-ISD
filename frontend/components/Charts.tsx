import { trendBars } from "@/lib/sample-data";

export function TrendChart({ title, note = "未接入可验证的时间序列；曲线仅用于展示页面布局，不代表真实趋势。" }: { title: string; note?: string }) {
  return (
    <section className="card chart-card">
      <div className="card-heading">
        <h3>{title}</h3>
        <span className="chart-tag">布局示意</span>
      </div>
      <div className="chart" aria-label={title}>
        {trendBars.map((height, index) => (
          <span key={index} style={{ height: `${height}%` }} />
        ))}
      </div>
      <p className="fine">{note}</p>
    </section>
  );
}

export function Bars({
  title,
  items,
  note = "图形仅按页面展示需要进行归一化，不代表模型输出或投资结论。"
}: {
  title: string;
  items: Array<{ label: string; value: number; display?: string }>;
  note?: string;
}) {
  return (
    <section className="card chart-card">
      <div className="card-heading">
        <h3>{title}</h3>
        <span className="chart-tag">证据优先</span>
      </div>
      <div className="bars">
        {items.map((item, index) => (
          <div className="bar" key={`${item.label}-${index}`}>
            <span>{item.label}</span>
            <span className="track">
              <span className="fill" style={{ width: `${Math.max(4, Math.min(100, item.value))}%` }} />
            </span>
            <span>{item.display ?? item.value}</span>
          </div>
        ))}
      </div>
      <p className="fine">{note}</p>
    </section>
  );
}
