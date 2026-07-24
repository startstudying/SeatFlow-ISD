export function MetricCard({
  icon,
  label,
  value,
  hint,
  accent = "accent-blue"
}: {
  icon: string;
  label: string;
  value: string;
  hint: string;
  accent?: "accent-blue" | "accent-green" | "accent-purple" | "accent-orange";
}) {
  return (
    <section className="card metric">
      <div className={`orb ${accent}`}>{icon}</div>
      <div>
        <div className="muted">{label}</div>
        <div className={`metric-value ${accent}`}>{value}</div>
        <p className="fine">{hint}</p>
      </div>
    </section>
  );
}
