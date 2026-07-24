import type { GraphSnapshot } from "@/lib/api";

const positions = [
  { left: "47%", top: "43%" },
  { left: "17%", top: "21%" },
  { left: "72%", top: "20%" },
  { left: "14%", top: "65%" },
  { left: "78%", top: "64%" },
  { left: "39%", top: "73%" },
  { left: "61%", top: "75%" },
  { left: "40%", top: "14%" }
];

const nodeNames: Record<string, string> = {
  sample_institution_a: "机构席位甲",
  sample_arbitrage_b: "机构席位乙",
  sample_daily_deviation: "日内偏离",
  sample_daily_deviation_reason: "日内偏离",
  large_cap: "大盘风格",
  banking: "银行",
  new_energy: "新能源",
  ai_compute: "人工智能算力"
};

const typeNames: Record<string, string> = {
  seat: "席位",
  institution: "机构席位",
  stock: "股票",
  industry: "行业",
  concept: "概念",
  event_reason: "事件原因",
  reason: "事件原因"
};

const relationNames: Record<string, string> = {
  buys: "买入",
  sells: "卖出",
  belongs_to: "归属",
  belongs_to_industry: "所属行业",
  belongs_to_concept: "所属概念",
  triggered_by: "触发原因",
  co_moves: "联动"
};

export function displayNodeName(value: string | undefined): string {
  if (!value) {
    return "未知节点";
  }
  return nodeNames[value] ?? value;
}

export function displayNodeType(value: string | undefined): string {
  return typeNames[value ?? ""] ?? value ?? "未知类型";
}

export function displayRelation(value: string | undefined): string {
  return relationNames[value ?? ""] ?? value ?? "未知关系";
}

export function edgeWidthPercent(weight: number | undefined): string {
  const value = typeof weight === "number" && Number.isFinite(weight) ? Math.abs(weight) : 0.3;
  const normalized = value <= 1 ? value : Math.log10(value + 1) / 8;
  const bounded = Math.min(1, Math.max(0.1, normalized));
  return `${18 + Math.round(bounded * 38)}%`;
}

function nodeTone(nodeType: string | undefined): string {
  if (nodeType === "stock") {
    return "node-stock";
  }
  if (nodeType === "industry") {
    return "node-industry";
  }
  if (nodeType === "concept") {
    return "node-concept";
  }
  if (nodeType === "event_reason" || nodeType === "reason") {
    return "node-reason";
  }
  return "node-seat";
}

export function GraphCanvas({
  graph,
  title = "动态资金流图",
  detail = false
}: {
  graph: GraphSnapshot | null;
  title?: string;
  detail?: boolean;
}) {
  const nodes = graph?.graph?.nodes?.slice(0, detail ? 8 : 7) ?? [];
  const edges = graph?.graph?.edges?.slice(0, 8) ?? [];

  return (
    <section className={`card graph-card ${detail ? "graph-detail" : ""}`}>
      <div className="card-heading graph-heading">
        <div>
          <h3>{title}</h3>
          <p className="fine">由接口图快照渲染；边线宽度仅用于保持可读性。</p>
        </div>
        <div className="graph-legend" aria-label="节点图例">
          <span className="legend-seat">席位</span>
          <span className="legend-stock">股票</span>
          <span className="legend-industry">行业</span>
          <span className="legend-concept">概念</span>
        </div>
      </div>
      <div className="graph-surface">
        {edges.map((edge, index) => (
          <span
            aria-label={`${displayRelation(edge.relation)}关系`}
            className={`edge edge-${index % 4}`}
            key={`${edge.source_id}-${edge.target_id}-${index}`}
            style={{
              left: `${12 + index * 8}%`,
              top: `${33 + (index % 4) * 12}%`,
              width: edgeWidthPercent(edge.weight),
              transform: `rotate(${index % 2 === 0 ? 18 : -23}deg)`
            }}
          />
        ))}
        {nodes.map((node, index) => (
          <div
            className={`node ${nodeTone(node.node_type)}`}
            key={`${node.node_type}-${node.node_id}-${index}`}
            style={{ left: positions[index]?.left, top: positions[index]?.top }}
            title={`${displayNodeType(node.node_type)}：${displayNodeName(node.node_id)}`}
          >
            <span>{displayNodeName(node.node_id)}</span>
            <small>{displayNodeType(node.node_type)}</small>
          </div>
        ))}
        {!nodes.length && <p className="graph-empty">暂未返回图节点，图区域保持为空以避免生成模拟关系。</p>}
      </div>
    </section>
  );
}
