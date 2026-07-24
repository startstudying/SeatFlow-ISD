import type { EvidenceMeta } from "@/lib/api";

const dataScopeLabels: Record<EvidenceMeta["data_scope"], string> = {
  real: "真实数据",
  sample: "示例数据",
  synthetic: "合成数据",
  none: "暂无数据",
  unknown: "范围未知"
};

const seatScopeLabels: Record<EvidenceMeta["seat_detail_scope"], string> = {
  complete_seat: "完整席位",
  institutional_only: "仅机构席位",
  synthetic: "合成席位",
  none: "暂无席位明细",
  unknown: "席位范围未知"
};

const modelStatusLabels: Record<EvidenceMeta["model_validity_status"], string> = {
  valid: "模型有效",
  invalid: "模型无效",
  not_available: "模型未接入",
  degraded: "模型已降级",
  smoke_only: "仅工程检查",
  unknown: "模型状态未知"
};

const riskLabels: Record<string, string> = {
  api_contract_not_verified: "接口契约未验证",
  api_request_failed: "接口请求失败",
  api_returned_no_data: "接口未返回数据",
  data_audit_not_verified: "数据审计未验证",
  explicit_ui_stub_not_real_data: "界面示例不代表真实数据",
  model_id_not_configured: "默认模型未配置",
  run_id_not_configured: "默认实验未配置",
  data_scope_sample: "示例数据范围",
  data_scope_synthetic: "合成数据范围",
  data_scope_none: "暂无数据范围"
};

export function EvidenceStrip({ meta }: { meta: EvidenceMeta }) {
  const isReal =
    meta.source === "api" &&
    meta.data_scope === "real" &&
    meta.seat_detail_scope === "complete_seat" &&
    meta.model_validity_status === "valid" &&
    meta.lineage_hash !== null &&
    meta.risk_flags.length === 0;
  const riskFlags = meta.risk_flags.length ? meta.risk_flags : ["无风险标记"];

  return (
    <section className="evidence-strip" aria-label="数据证据状态">
      <div className="evidence-title">
        <i className={isReal ? "status-dot status-good" : "status-dot status-warn"} />
        <span>{isReal ? "证据链完整" : "证据链待补全"}</span>
      </div>
      <div className="badge-row">
        <span className={isReal ? "badge good" : "badge warn"} title={`原始值：${meta.data_scope}`}>
          数据范围：{dataScopeLabels[meta.data_scope]}
        </span>
        <span className={isReal ? "badge good" : "badge warn"} title={`原始值：${meta.seat_detail_scope}`}>
          席位范围：{seatScopeLabels[meta.seat_detail_scope]}
        </span>
        <span className={isReal ? "badge good" : "badge warn"} title={`原始值：${meta.model_validity_status}`}>
          模型状态：{modelStatusLabels[meta.model_validity_status]}
        </span>
        <span className="badge" title={meta.lineage_hash ?? "缺失"}>
          谱系标识：{meta.lineage_hash ? `${meta.lineage_hash.slice(0, 12)}...` : "缺失"}
        </span>
        <span className="badge">来源：{meta.source === "api" ? "接口返回" : "界面示例"}</span>
        {riskFlags.map((flag) => (
          <span className={flag === "无风险标记" ? "badge good" : "badge warn"} key={flag} title={`原始值：${flag}`}>
            风险：{riskLabels[flag] ?? flag}
          </span>
        ))}
      </div>
    </section>
  );
}
