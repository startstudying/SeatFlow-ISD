import type { GraphSnapshot, MarketPayload } from "./api";

export const sampleGraph: GraphSnapshot = {
  data_scope: "synthetic",
  lineage_hash: "local-ui-stub",
  risk_flags: ["explicit_ui_stub_not_real_data"],
  model_validity_status: "smoke_only",
  manifest: {
    row_count: 7,
    content_sha256: "local-ui-stub"
  },
  graph: {
    as_of_date: "2026-01-15",
    nodes: [
      { node_type: "seat", node_id: "华泰证券深圳益田路" },
      { node_type: "stock", node_id: "600519.SH" },
      { node_type: "stock", node_id: "300750.SZ" },
      { node_type: "industry", node_id: "新能源" },
      { node_type: "concept", node_id: "AI算力" },
      { node_type: "event_reason", node_id: "业绩预增" }
    ],
    edges: [
      { source_id: "华泰证券深圳益田路", target_id: "600519.SH", relation: "buys", weight: 0.78 },
      { source_id: "华泰证券深圳益田路", target_id: "300750.SZ", relation: "buys", weight: 0.62 },
      { source_id: "300750.SZ", target_id: "新能源", relation: "belongs_to", weight: 0.54 },
      { source_id: "600519.SH", target_id: "AI算力", relation: "co_moves", weight: 0.31 }
    ]
  }
};

export const sampleMarket: MarketPayload = {
  summary: {
    provider: "sample",
    snapshot_count: 2,
    datasets: [
      { store: "processed", dataset: "daily_bar" },
      { store: "graphs", dataset: "hetero_graph" }
    ],
    data_scope: "synthetic",
    lineage_hash: "local-ui-stub",
    risk_flags: ["explicit_ui_stub_not_real_data"],
    model_validity_status: "smoke_only"
  },
  graph: sampleGraph
};

export const trendBars = [42, 55, 58, 49, 45, 52, 63, 61, 76, 88, 70, 67, 74, 82];
