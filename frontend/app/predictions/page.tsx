import { Bars, TrendChart } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { EvidenceStrip } from "@/components/EvidenceStrip";
import { GraphCanvas } from "@/components/GraphCanvas";
import { TopBar } from "@/components/TopBar";
import { fetchPredictionState } from "@/lib/api";
import { maybeStubState } from "@/lib/page-state";
import { sampleGraph } from "@/lib/sample-data";

export default async function Predictions({
  searchParams
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const state = (await maybeStubState(searchParams, sampleGraph)) ?? (await fetchPredictionState());
  const message = state.kind === "empty" || state.kind === "error" ? state.message : "尚未取得可用于展示的真实预测契约。";

  return (
    <>
      <TopBar title="股票预测分析" icon="↗" />
      <EvidenceStrip meta={state.meta} />
      <div className="page-grid prediction-page">
        <section className="prediction-warning">
          <span>预测服务未就绪</span>
          <p>缺少有效模型、完整席位图和预测谱系，禁止展示上涨概率、收益、目标价或买卖评级。</p>
          <small>{message}</small>
        </section>

        <div className="grid prediction-top">
          <section className="card prediction-hero">
            <div className="stock-placeholder">
              <span className="eyebrow">目标标的</span>
              <h3>--</h3>
              <strong>等待模型服务返回</strong>
              <p>代码、名称、价格与涨跌幅均不使用原型演示值填充。</p>
              <div className="stock-meta"><span>数据状态：待验证</span><span>更新时间：--</span></div>
            </div>
            <div className="prediction-ring pending-ring">
              <div><strong>--</strong><span>预测概率</span><small>待验证</small></div>
            </div>
          </section>

          <DataTable
            title="重点注意力因子"
            columns={["因子", "类型", "权重", "影响"]}
            rows={[
              ["席位影响", "待接入", "--", "待验证"],
              ["行业资金流", "待接入", "--", "待验证"],
              ["机构共现", "待接入", "--", "待验证"],
              ["事件原因", "待接入", "--", "待验证"]
            ]}
            note="注意力因子必须绑定实际预测、模型版本和谱系标识。"
          />
        </div>

        <div className="grid three prediction-middle">
          <TrendChart title="价格与成交量（近30日）" note="无真实标的和行情契约时，不展示价格、成交量或收益曲线。" />
          <Bars
            title="预测置信度走势"
            items={[
              { label: "1日", value: 0, display: "--" },
              { label: "3日", value: 0, display: "--" },
              { label: "5日", value: 0, display: "--" }
            ]}
            note="未返回真实预测分布，所有置信度数值保持为空。"
          />
          <GraphCanvas graph={null} title="资金传播解释子图" />
        </div>

        <div className="grid prediction-bottom">
          <section className="card explanation-card">
            <div className="card-heading"><h3>模型解释</h3><span className="chart-tag">未放行</span></div>
            <ul className="insight-list">
              <li><i className="accent-blue" />资金驱动：等待真实图谱与边权</li>
              <li><i className="accent-green" />机构共识：等待共现席位统计</li>
              <li><i className="accent-purple" />行业景气：等待行业聚合契约</li>
              <li><i className="accent-orange" />风险提示：等待模型和数据审计</li>
            </ul>
          </section>
          <section className="card gate-card">
            <div className="card-heading"><h3>预测放行条件</h3><span className="chart-tag">保守关闭</span></div>
            <ol>
              <li>返回真实数据和完整席位明细。</li>
              <li>模型有效且存在可追溯谱系标识。</li>
              <li>风险标记为空，并有真实盲测或回测证据。</li>
            </ol>
          </section>
        </div>
      </div>
    </>
  );
}
