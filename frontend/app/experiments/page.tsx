import { Bars, TrendChart } from "@/components/Charts";
import { DataTable } from "@/components/DataTable";
import { EvidenceStrip } from "@/components/EvidenceStrip";
import { MetricCard } from "@/components/MetricCard";
import { TopBar } from "@/components/TopBar";
import { fetchExperimentState } from "@/lib/api";
import { maybeStubState } from "@/lib/page-state";
import { sampleGraph } from "@/lib/sample-data";

export default async function Experiments({
  searchParams
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const state = (await maybeStubState(searchParams, sampleGraph)) ?? (await fetchExperimentState());
  const message = state.kind === "empty" || state.kind === "error" ? state.message : "尚未取得可验证的实验运行记录。";

  return (
    <>
      <TopBar title="实验中心" icon="◫" searchHint="搜索实验、模型、指标..." />
      <EvidenceStrip meta={state.meta} />
      <div className="page-grid experiments-page">
        <section className="prediction-warning experiment-warning">
          <span>真实实验结果未放行</span>
          <p>未提供已冻结的数据切分、训练谱系与盲测证据，因此 AUC、RankIC、夏普比率等指标保持为空。</p>
          <small>{message}</small>
        </section>

        <div className="grid metrics metrics-six">
          <MetricCard icon="◈" label="模型版本" value="待登记" hint="等待模型注册信息" />
          <MetricCard icon="▦" label="数据集窗口" value="待确认" hint="等待冻结的数据切分" accent="accent-purple" />
          <MetricCard icon="A" label="最佳 AUC" value="--" hint="不展示模拟指标" accent="accent-green" />
          <MetricCard icon="R" label="最佳 RankIC" value="--" hint="不展示模拟指标" accent="accent-blue" />
          <MetricCard icon="S" label="夏普比率" value="--" hint="需要真实回测证据" accent="accent-purple" />
          <MetricCard icon="!" label="训练状态" value="待验证" hint="工程形状检查不等于训练完成" accent="accent-orange" />
        </div>

        <div className="grid experiments-focus">
          <DataTable
            title="模型对比"
            columns={["模型", "AUC", "F1", "RankIC", "状态"]}
            rows={[
              ["SeatFlow", "--", "--", "--", "待验证"],
              ["长短期记忆网络", "--", "--", "--", "待验证"],
              ["图注意力网络", "--", "--", "--", "待验证"],
              ["时间异构图网络", "--", "--", "--", "待验证"]
            ]}
            note="模型对比仅在使用相同冻结切分与评估协议后展示。"
          />
          <TrendChart title="训练进度" note="尚未取得可验证的训练日志和指标序列；曲线只复刻仪表盘布局。" />
        </div>

        <div className="grid three experiments-middle">
          <Bars
            title="消融验证"
            items={[
              { label: "完整模型", value: 0, display: "--" },
              { label: "移除席位节点", value: 0, display: "--" },
              { label: "移除时间衰减", value: 0, display: "--" },
              { label: "移除概念节点", value: 0, display: "--" }
            ]}
            note="未完成多组真实实验，消融指标保持为空。"
          />
          <section className="card pipeline-card">
            <div className="card-heading"><h3>运行日志与流水线状态</h3><span className="chart-tag">工程状态</span></div>
            <ul className="pipeline-list">
              <li><i className="status-dot status-good" /><span>接口配置</span><strong>已配置</strong></li>
              <li><i className="status-dot status-good" /><span>图谱工程检查</span><strong>已完成</strong></li>
              <li><i className="status-dot status-warn" /><span>真实数据审计</span><strong>待验证</strong></li>
              <li><i className="status-dot status-warn" /><span>模型训练与盲测</span><strong>待验证</strong></li>
            </ul>
            <p className="fine">“已完成”仅指本地工程检查，不构成真实训练或回测结论。</p>
          </section>
          <section className="card config-card">
            <div className="card-heading"><h3>实验配置</h3><span className="chart-tag">待冻结</span></div>
            <dl className="config-list">
              <div><dt>节点类型</dt><dd>席位、股票、行业、概念</dd></div>
              <div><dt>边类型</dt><dd>买入、卖出、归属、触发</dd></div>
              <div><dt>评估协议</dt><dd>等待 protocol.lock</dd></div>
              <div><dt>随机种子</dt><dd>等待训练清单</dd></div>
            </dl>
          </section>
        </div>

        <section className="card workflow-card">
          <div className="card-heading"><h3>模型产物与工作流</h3><span className="chart-tag">研究闭环</span></div>
          <div className="workflow-steps">
            <span>数据契约</span><i>→</i><span>图构建</span><i>→</i><span>训练框架</span><i>→</i><span>实验追踪</span><i>→</i><span>模型登记</span><i>→</i><span>服务部署</span>
          </div>
          <p className="fine">训练框架可使用 PyTorch Geometric，实验追踪可使用 MLflow；所有产物必须绑定真实数据、模型与运行谱系。</p>
        </section>
      </div>
    </>
  );
}
