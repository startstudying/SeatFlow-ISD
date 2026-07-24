import { TrendChart } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { displayNodeName, displayNodeType, GraphCanvas } from "@/components/GraphCanvas";
import { MetricCard } from "@/components/MetricCard";
import { StateFrame } from "@/components/StateFrame";
import { TopBar } from "@/components/TopBar";
import { fetchMarketState } from "@/lib/api";
import { maybeStubState } from "@/lib/page-state";
import { sampleMarket } from "@/lib/sample-data";

const datasetNames: Record<string, string> = {
  daily_bar: "日线行情",
  hetero_graph: "异构图快照"
};

const storeNames: Record<string, string> = {
  processed: "处理数据",
  graphs: "图谱数据"
};

export default async function MarketOverview({
  searchParams
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const state = (await maybeStubState(searchParams, sampleMarket)) ?? (await fetchMarketState());

  return (
    <>
      <TopBar title="市场总览" icon="◇" />
      <StateFrame state={state}>
        {(data, degradedReason) => {
          const nodes = data.graph?.graph?.nodes ?? [];
          const nodeCount = nodes.length;
          const edgeCount = data.graph?.graph?.edges?.length ?? 0;
          const stockRows = nodes
            .filter((node) => node.node_type === "stock")
            .slice(0, 4)
            .map((node) => [displayNodeName(node.node_id), displayNodeType(node.node_type), "待验证"]);
          const datasetRows = data.summary.datasets.map((item) => {
            const store = String(item.store ?? "unknown");
            const dataset = String(item.dataset ?? "unknown");
            return [storeNames[store] ?? store, datasetNames[dataset] ?? dataset, data.summary.provider === "sample" ? "示例源" : data.summary.provider];
          });

          return (
            <div className="page-grid dashboard-page">
              {degradedReason && <section className="degraded-banner">{degradedReason}</section>}
              <div className="grid metrics">
                <MetricCard icon="▦" label="数据快照" value={String(data.summary.snapshot_count)} hint="来自已返回的快照汇总" />
                <MetricCard icon="◎" label="图谱节点" value={String(nodeCount)} hint="来自当前图快照" accent="accent-green" />
                <MetricCard icon="⇢" label="传播关系" value={String(edgeCount)} hint="来自当前图边集合" accent="accent-purple" />
                <MetricCard icon="!" label="模型结论" value="待验证" hint="缺少有效模型与完整证据链" accent="accent-orange" />
              </div>

              <div className="grid two dashboard-focus">
                <GraphCanvas graph={data.graph} title="动态资金流图" />
                <DataTable
                  title="重点观察标的"
                  columns={["标的", "节点类型", "预测状态"]}
                  rows={stockRows.length ? stockRows : [["--", "等待图快照", "待验证"]]}
                  note="未取得真实有效模型时，不展示上涨概率、目标价或买卖评级。"
                />
              </div>

              <div className="grid three dashboard-bottom">
                <TrendChart title="席位活跃趋势" note="未接入真实时序聚合；图形仅复刻研究看板布局。" />
                <section className="card heat-card">
                  <div className="card-heading">
                    <h3>行业热力图</h3>
                    <span className="chart-tag">待接入</span>
                  </div>
                  <div className="heat-grid" aria-label="行业热力图布局示意">
                    <span>银行</span>
                    <span>新能源</span>
                    <span>人工智能</span>
                    <span>医药</span>
                    <span>消费</span>
                    <span>半导体</span>
                    <span>证券</span>
                    <span>电力设备</span>
                  </div>
                  <p className="fine">行业聚合接口尚未返回，此处不表达任何行业强弱结论。</p>
                </section>
                <section className="card insight-card">
                  <div className="card-heading">
                    <h3>研究说明</h3>
                    <span className="chart-tag">证据优先</span>
                  </div>
                  <ul className="insight-list">
                    <li><i className="accent-blue" />图谱关系来自当前接口快照</li>
                    <li><i className="accent-orange" />完整席位明细尚未通过验证</li>
                    <li><i className="accent-purple" />真实预测与收益结论保持关闭</li>
                  </ul>
                  <p className="fine">仅当数据范围、席位范围、模型有效性和谱系同时完整时，页面才会放行真实预测结果。</p>
                </section>
              </div>

              <DataTable title="数据接入概览" columns={["存储层", "数据集", "数据来源"]} rows={datasetRows.length ? datasetRows : [["--", "暂无数据集", "--"]]} />
            </div>
          );
        }}
      </StateFrame>
    </>
  );
}
