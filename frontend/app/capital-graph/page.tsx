import { Bars } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { displayNodeName, displayNodeType, displayRelation, GraphCanvas } from "@/components/GraphCanvas";
import { StateFrame } from "@/components/StateFrame";
import { TopBar } from "@/components/TopBar";
import { fetchCapitalState } from "@/lib/api";
import { maybeStubState } from "@/lib/page-state";
import { sampleGraph } from "@/lib/sample-data";

function displayWeight(weight: number | undefined): string {
  return typeof weight === "number" && Number.isFinite(weight) ? weight.toLocaleString("zh-CN", { maximumFractionDigits: 3 }) : "--";
}

function relativeWidth(weight: number | undefined, maxWeight: number): number {
  if (typeof weight !== "number" || !Number.isFinite(weight) || maxWeight <= 0) {
    return 0;
  }
  return Math.max(4, Math.round((Math.log10(Math.abs(weight) + 1) / Math.log10(maxWeight + 1)) * 100));
}

export default async function CapitalGraph({
  searchParams
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const state = (await maybeStubState(searchParams, sampleGraph)) ?? (await fetchCapitalState());

  return (
    <>
      <TopBar title="资金传播图" icon="⌘" />
      <StateFrame state={state}>
        {(data, degradedReason) => {
          const nodes = data.graph?.nodes ?? [];
          const edges = data.graph?.edges ?? [];
          const focusNode = nodes[0];
          const maxWeight = Math.max(...edges.map((edge) => Math.abs(edge.weight ?? 0)), 0);
          const edgeRows = edges.slice(0, 5).map((edge) => [
            `${displayNodeName(edge.source_id)} → ${displayNodeName(edge.target_id)}`,
            displayRelation(edge.relation),
            displayWeight(edge.weight)
          ]);

          return (
            <div className="page-grid capital-page">
              {degradedReason && <section className="degraded-banner">{degradedReason}</section>}
              <section className="toolbar card">
                <div className="segmented" aria-label="时间范围">
                  <button className="active" type="button">当日</button>
                  <button type="button">近7日</button>
                  <button type="button">近30日</button>
                </div>
                <div className="toolbar-controls">
                  <span className="control-chip">日期：{data.graph?.as_of_date ?? "--"}</span>
                  <span className="control-chip">节点类型：全部</span>
                  <span className="threshold">展示阈值 <i><b /></i> 由接口控制</span>
                </div>
              </section>

              <div className="grid capital-focus">
                <GraphCanvas graph={data} title="资金传播图" detail />
                <section className="card inspector-card">
                  <div className="card-heading">
                    <h3>节点检查器</h3>
                    <span className="chart-tag">图快照</span>
                  </div>
                  <div className="inspector-hero">
                    <span className="node-badge">{displayNodeType(focusNode?.node_type)}</span>
                    <div>
                      <strong>{displayNodeName(focusNode?.node_id)}</strong>
                      <p>{focusNode ? "当前选中图谱首节点" : "等待图节点返回"}</p>
                    </div>
                  </div>
                  <dl className="inspector-list">
                    <div><dt>节点类型</dt><dd>{displayNodeType(focusNode?.node_type)}</dd></div>
                    <div><dt>图谱日期</dt><dd>{data.graph?.as_of_date ?? "--"}</dd></div>
                    <div><dt>节点总数</dt><dd>{nodes.length}</dd></div>
                    <div><dt>关系总数</dt><dd>{edges.length}</dd></div>
                    <div><dt>模型注意力</dt><dd>未验证</dd></div>
                  </dl>
                  <p className="fine">暂未取得可解释模型输出，不展示胜率、收益、风险评分或注意力因子。</p>
                </section>
              </div>

              <div className="grid three capital-bottom">
                <DataTable
                  title="传播路径"
                  columns={["路径", "关系", "图边权重"]}
                  rows={edgeRows.length ? edgeRows : [["--", "暂无关系", "--"]]}
                  note="图边权重来自当前快照，只用于关系展示，不等同模型注意力或投资影响。"
                />
                <Bars
                  title="边权重拆解"
                  items={edges.slice(0, 5).map((edge, index) => ({
                    label: displayRelation(edge.relation) || `关系${index + 1}`,
                    value: relativeWidth(edge.weight, maxWeight),
                    display: "相对值"
                  }))}
                  note="按图边权进行对数归一化，仅用于可读性，不构成模型解释。"
                />
                <section className="card cooccurrence-card">
                  <div className="card-heading">
                    <h3>共现席位</h3>
                    <span className="chart-tag">待接入</span>
                  </div>
                  <div className="empty-visual">⌁</div>
                  <strong>暂无共现席位契约</strong>
                  <p className="muted">后端需要返回共现统计、观察窗口和谱系标识后才会展示结果。</p>
                </section>
              </div>
            </div>
          );
        }}
      </StateFrame>
    </>
  );
}
