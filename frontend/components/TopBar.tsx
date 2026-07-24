export function TopBar({ title, icon, searchHint = "搜索股票、席位、行业..." }: { title: string; icon: string; searchHint?: string }) {
  const tradeDate = process.env.NEXT_PUBLIC_SEATFLOW_TRADE_DATE ?? "2026-01-15";

  return (
    <header className="topbar">
      <div className="title">
        <span className="title-icon" aria-hidden="true">
          {icon}
        </span>
        <div>
          <p className="eyebrow">SeatFlow 研究工作台</p>
          <h2>{title}</h2>
        </div>
      </div>
      <div className="chrome">
        <span className="date-chip">◷ {tradeDate}</span>
        <div className="search" role="search" aria-label="搜索">
          <span aria-hidden="true">⌕</span>
          <span>{searchHint}</span>
        </div>
        <span className="notice" aria-label="3 条通知">
          ♧<b>3</b>
        </span>
        <span className="researcher">
          <i>量</i>
          <span>量化研究员</span>
        </span>
      </div>
    </header>
  );
}
