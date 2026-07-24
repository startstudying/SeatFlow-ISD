"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, Database, FileText, RefreshCw, Save, ServerCog, Settings2, ShieldCheck, Users } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useSettingsMutations, useSettingsQuery } from "@/features/settings/query";
import { settingsQueryKey } from "@/features/settings/query";
import type { DataSourceConfig, ExperimentConfig, GeneralSettingsInput, ModelConfig, SettingsPayload, SettingsSection } from "@/features/settings/types";
import { settingsSectionLabels } from "@/features/settings/types";
import styles from "./SettingsPage.module.css";

const sections = Object.keys(settingsSectionLabels) as SettingsSection[];
const auditPageSize = 6;

const generalSchema = z.object({
  systemName: z.string().trim().min(2, "系统名称至少需要 2 个字符").max(64, "系统名称不能超过 64 个字符"),
  timezone: z.string().trim().min(1, "请选择时区"),
  market: z.string().trim().min(1, "请选择默认市场"),
  retentionDays: z.coerce.number().int("保留天数必须为整数").min(7, "最少保留 7 天").max(3650, "最多保留 3650 天")
});

const sourceSchema = z.object({
  displayName: z.string().trim().min(2, "名称至少需要 2 个字符").max(128),
  provider: z.string().trim().min(2, "请输入数据源标识").max(64),
  endpoint: z.url("请输入有效的 HTTPS/HTTP 地址"),
  apiKey: z.string().max(4096, "密钥长度不能超过 4096 个字符").optional()
});

type GeneralForm = z.input<typeof generalSchema>;
type SourceForm = z.input<typeof sourceSchema>;
type Notice = { tone: "success" | "error"; text: string } | null;

export function SettingsPage() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const query = useSettingsQuery();
  const [notice, setNotice] = useState<Notice>(null);
  const requestedTab = searchParams.get("tab");
  const activeSection = sections.includes(requestedTab as SettingsSection) ? requestedTab as SettingsSection : "general";

  function selectSection(section: SettingsSection) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", section);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>SeatFlow / 系统管理</p>
          <h1>系统设置</h1>
          <p className={styles.subtitle}>集中管理系统参数、数据源、模型服务、实验环境、权限和审计记录。</p>
        </div>
        <div className={styles.headerActions}>
          <ModeBadge mode={query.data?.mode} />
          <button className={styles.outlineButton} disabled={query.isFetching} onClick={() => void query.refetch()} type="button">
            <RefreshCw size={15} /> {query.isFetching ? "刷新中" : "刷新状态"}
          </button>
        </div>
      </header>

      <nav aria-label="系统设置二级导航" className={styles.sectionTabs}>
        {sections.map((section) => (
          <button className={activeSection === section ? styles.activeTab : undefined} key={section} onClick={() => selectSection(section)} type="button">
            {settingsSectionLabels[section]}
          </button>
        ))}
      </nav>

      {notice && <NoticeBanner notice={notice} onClose={() => setNotice(null)} />}
      {query.isLoading && <LoadingState />}
      {query.isError && <ErrorState message={query.error.message} onRetry={() => void query.refetch()} />}
      {query.data && <SettingsContent activeSection={activeSection} data={query.data} onNotice={setNotice} />}
    </main>
  );
}

function SettingsContent({ activeSection, data, onNotice }: { activeSection: SettingsSection; data: SettingsPayload; onNotice: (notice: Notice) => void }) {
  switch (activeSection) {
    case "general":
      return <GeneralSection data={data} onNotice={onNotice} />;
    case "data-sources":
      return <DataSourcesSection data={data} onNotice={onNotice} />;
    case "model":
      return <ModelsSection data={data} onNotice={onNotice} />;
    case "experiment":
      return <ExperimentsSection data={data} onNotice={onNotice} />;
    case "monitoring":
      return <MonitoringSection data={data} />;
    case "users":
      return <UsersSection data={data} />;
    case "audit":
      return <AuditSection data={data} />;
  }
}

function GeneralSection({ data, onNotice }: { data: SettingsPayload; onNotice: (notice: Notice) => void }) {
  const general = data.settings.find((item) => item.key === "general");
  const defaultValues = useMemo<GeneralForm>(() => ({
    systemName: readString(general?.value.systemName, "SeatFlow 科研量化平台"),
    timezone: readString(general?.value.timezone, "Asia/Shanghai"),
    market: readString(general?.value.market, "A 股（中国内地）"),
    retentionDays: readNumber(general?.value.retentionDays, 180)
  }), [general]);
  const form = useForm<GeneralForm>({ resolver: zodResolver(generalSchema), defaultValues });
  const mutations = useSettingsMutations();
  const queryClient = useQueryClient();

  useEffect(() => form.reset(defaultValues), [defaultValues, form]);

  async function onSubmit(values: GeneralForm) {
    try {
      await mutations.saveGeneral.mutateAsync({ input: generalSchema.parse(values) as GeneralSettingsInput, version: general?.version ?? 0 });
      onNotice({ tone: "success", text: "通用设置已保存，并记录了审计事件。" });
    } catch (error) {
      onNotice({ tone: "error", text: toErrorMessage(error, "保存失败，请检查版本或服务状态。") });
    }
  }

  return (
    <div className={styles.contentGrid}>
      <section className={`${styles.panel} ${styles.widePanel}`}>
        <PanelHeading icon={<Settings2 size={17} />} title="通用设置" description="提交前进行客户端校验；真实接口模式下还会执行乐观锁和幂等检查。" />
        <form className={styles.formGrid} onSubmit={form.handleSubmit(onSubmit)}>
          <Field label="系统名称" error={form.formState.errors.systemName?.message}>
            <input {...form.register("systemName")} />
          </Field>
          <Field label="默认时区" error={form.formState.errors.timezone?.message}>
            <select {...form.register("timezone")}><option value="Asia/Shanghai">Asia/Shanghai（中国标准时间）</option><option value="UTC">UTC</option></select>
          </Field>
          <Field label="默认市场" error={form.formState.errors.market?.message}>
            <select {...form.register("market")}><option value="A 股（中国内地）">A 股（中国内地）</option><option value="研究沙盒">研究沙盒</option></select>
          </Field>
          <Field label="审计保留天数" error={form.formState.errors.retentionDays?.message}>
            <input min="7" max="3650" type="number" {...form.register("retentionDays")} />
          </Field>
          <div className={styles.formActions}><button className={styles.primaryButton} disabled={mutations.saveGeneral.isPending} type="submit"><Save size={15} /> {mutations.saveGeneral.isPending ? "保存中" : "保存设置"}</button></div>
        </form>
      </section>
      <DangerZone onComplete={() => { queryClient.clear(); onNotice({ tone: "success", text: "本地查询缓存已清除，页面将重新加载受控数据。" }); }} />
    </div>
  );
}

function DataSourcesSection({ data, onNotice }: { data: SettingsPayload; onNotice: (notice: Notice) => void }) {
  const selected = data.dataSources[0];
  const defaults = useMemo<SourceForm>(() => ({
    displayName: selected?.displayName ?? "新增数据源",
    provider: selected?.provider ?? "akshare",
    endpoint: selected?.endpoint ?? "https://akshare.akfamily.xyz",
    apiKey: ""
  }), [selected]);
  const form = useForm<SourceForm>({ resolver: zodResolver(sourceSchema), defaultValues: defaults });
  const mutations = useSettingsMutations();

  useEffect(() => form.reset(defaults), [defaults, form]);

  async function onSubmit(values: SourceForm) {
    try {
      const parsed = sourceSchema.parse(values);
      await mutations.saveDataSource.mutateAsync({
        id: selected?.id ?? "custom-source",
        input: { ...parsed, metadata: selected?.metadata ?? { schedule: "手动" }, expectedVersion: selected?.version ?? 0 }
      });
      onNotice({ tone: "success", text: "数据源配置已保存。密钥只会加密写入，页面不会回显明文。" });
      form.setValue("apiKey", "");
    } catch (error) {
      onNotice({ tone: "error", text: toErrorMessage(error, "数据源保存失败。") });
    }
  }

  return (
    <div className={styles.contentGrid}>
      <section className={styles.panel}>
        <PanelHeading icon={<Database size={17} />} title="已接入数据源" description="接口密钥永不从 API 回传；只显示配置状态或脱敏标识。" />
        {data.dataSources.length ? <div className={styles.sourceList}>{data.dataSources.map((source) => <SourceCard key={source.id} source={source} />)}</div> : <EmptyState message="暂未配置数据源。保存下方表单后会创建首个数据源。" />}
      </section>
      <section className={styles.panel}>
        <PanelHeading icon={<ShieldCheck size={17} />} title={selected ? `编辑：${selected.displayName}` : "新增数据源"} description="留空密钥字段表示保留已配置密钥；填写后将覆盖并加密保存。" />
        <form className={styles.formGrid} onSubmit={form.handleSubmit(onSubmit)}>
          <Field label="显示名称" error={form.formState.errors.displayName?.message}><input {...form.register("displayName")} /></Field>
          <Field label="数据提供方" error={form.formState.errors.provider?.message}><input {...form.register("provider")} /></Field>
          <Field label="服务地址" error={form.formState.errors.endpoint?.message}><input {...form.register("endpoint")} /></Field>
          <Field label="接口密钥（仅可写入）" hint={selected?.apiKeyMasked ?? "未配置"} error={form.formState.errors.apiKey?.message}><input autoComplete="off" placeholder="留空则不修改" type="password" {...form.register("apiKey")} /></Field>
          <div className={styles.formActions}><button className={styles.primaryButton} disabled={mutations.saveDataSource.isPending} type="submit"><Save size={15} /> {mutations.saveDataSource.isPending ? "保存中" : "保存数据源"}</button></div>
        </form>
      </section>
    </div>
  );
}

function ModelsSection({ data, onNotice }: { data: SettingsPayload; onNotice: (notice: Notice) => void }) {
  const model = data.models[0];
  const mutations = useSettingsMutations();
  async function save() {
    try {
      await mutations.saveModel.mutateAsync({ id: model?.id ?? "graph-model", input: { displayName: model?.displayName ?? "席位异构图模型", provider: model?.provider ?? "python-service", status: model?.status ?? "待验证", parameters: model?.parameters ?? { hiddenDim: 256 }, expectedVersion: model?.version ?? 0 } });
      onNotice({ tone: "success", text: "模型配置已保存，模型切换记录将进入审计日志。" });
    } catch (error) { onNotice({ tone: "error", text: toErrorMessage(error, "模型配置保存失败。") }); }
  }
  return <section className={styles.panel}><PanelHeading icon={<ServerCog size={17} />} title="模型配置" description="模型参数由 Java 服务托管；模型本身仍需通过数据、训练和盲测门槛。" /><ConfigTable rows={data.models.map((item) => [item.displayName, item.provider, item.status, stringify(item.parameters), `v${item.version}`])} columns={["模型", "服务", "状态", "参数", "版本"]} empty="暂未登记模型配置。" /><div className={styles.formActions}><button className={styles.primaryButton} disabled={mutations.saveModel.isPending} onClick={() => void save()} type="button"><Save size={15} /> {mutations.saveModel.isPending ? "保存中" : "保存当前模型配置"}</button></div></section>;
}

function ExperimentsSection({ data, onNotice }: { data: SettingsPayload; onNotice: (notice: Notice) => void }) {
  const experiment = data.experiments[0];
  const mutations = useSettingsMutations();
  async function save() {
    try {
      await mutations.saveExperiment.mutateAsync({ id: experiment?.id ?? "research-default", input: { name: experiment?.name ?? "研究基线实验", status: experiment?.status ?? "待验证", config: experiment?.config ?? { seed: 42 }, expectedVersion: experiment?.version ?? 0 } });
      onNotice({ tone: "success", text: "实验配置已保存。训练或回测结论仍需独立证据。" });
    } catch (error) { onNotice({ tone: "error", text: toErrorMessage(error, "实验配置保存失败。") }); }
  }
  return <section className={styles.panel}><PanelHeading icon={<FileText size={17} />} title="实验配置" description="此处只管理实验参数，不显示未经验证的模型结果或收益结论。" /><ConfigTable rows={data.experiments.map((item) => [item.name, item.status, stringify(item.config), `v${item.version}`])} columns={["实验", "状态", "配置", "版本"]} empty="暂未登记实验配置。" /><div className={styles.formActions}><button className={styles.primaryButton} disabled={mutations.saveExperiment.isPending} onClick={() => void save()} type="button"><Save size={15} /> {mutations.saveExperiment.isPending ? "保存中" : "保存实验配置"}</button></div></section>;
}

function MonitoringSection({ data }: { data: SettingsPayload }) {
  return <div className={styles.contentGrid}><section className={`${styles.panel} ${styles.widePanel}`}><PanelHeading icon={<ServerCog size={17} />} title="服务监控" description="来自统一业务 API 的最新服务状态；不向浏览器暴露内部服务凭据。" /><ConfigTable rows={data.services.map((item) => [item.serviceId, <StatusPill key={item.serviceId} value={item.status} />, formatDate(item.checkedAt), stringify(item.metrics), `v${item.version}`])} columns={["服务", "状态", "最近检查", "指标", "版本"]} empty="尚无服务监控记录。" /></section><StatusSummary services={data.services} /></div>;
}

function UsersSection({ data }: { data: SettingsPayload }) {
  return <div className={styles.contentGrid}><section className={styles.panel}><PanelHeading icon={<Users size={17} />} title="用户权限" description="RBAC 角色和权限由 Java API 管理。删除和权限变更必须经过后端授权与审计。" /><ConfigTable rows={data.users.map((item) => [item.displayName, item.id, item.roles.join("、") || "未分配", <StatusPill key={item.id} value={item.active ? "已启用" : "已停用"} />, `v${item.version}`])} columns={["用户", "标识", "角色", "状态", "版本"]} empty="尚无用户记录。" /></section><section className={styles.panel}><PanelHeading icon={<ShieldCheck size={17} />} title="角色与权限" description="角色权限为只读视图；生产变更须使用受审计的管理接口。" /><ConfigTable rows={data.roles.map((role) => [role.name, role.code, role.permissions.join("、") || "未授权"]) } columns={["角色", "代码", "权限"]} empty="尚无角色记录。" /></section></div>;
}

function AuditSection({ data }: { data: SettingsPayload }) {
  const [keyword, setKeyword] = useState("");
  const [page, setPage] = useState(1);
  const filtered = useMemo(() => data.auditLogs.filter((item) => `${item.actor} ${item.action} ${item.resourceType} ${item.resourceId}`.toLowerCase().includes(keyword.trim().toLowerCase())), [data.auditLogs, keyword]);
  const pageCount = Math.max(1, Math.ceil(filtered.length / auditPageSize));
  const visible = filtered.slice((page - 1) * auditPageSize, page * auditPageSize);
  return <section className={styles.panel}><PanelHeading icon={<FileText size={17} />} title="日志审计" description="每次受控写操作都携带 requestId 和幂等键，便于追踪与重放保护。" /><div className={styles.tableTools}><input aria-label="筛选审计日志" onChange={(event) => { setKeyword(event.target.value); setPage(1); }} placeholder="按操作人、资源或动作筛选" value={keyword} /><span>{filtered.length} 条记录</span></div><ConfigTable rows={visible.map((item) => [formatDate(item.createdAt), item.actor, item.action, `${item.resourceType}/${item.resourceId}`, item.requestId])} columns={["时间", "操作人", "动作", "资源", "请求标识"]} empty="未找到匹配的审计记录。" /><div className={styles.pagination}><button disabled={page <= 1} onClick={() => setPage((value) => value - 1)} type="button"><ChevronLeft size={15} />上一页</button><span>{page} / {pageCount}</span><button disabled={page >= pageCount} onClick={() => setPage((value) => value + 1)} type="button">下一页<ChevronRight size={15} /></button></div></section>;
}

function PanelHeading({ icon, title, description }: { icon: ReactNode; title: string; description: string }) {
  return <div className={styles.panelHeading}><span>{icon}</span><div><h2>{title}</h2><p>{description}</p></div></div>;
}

function Field({ label, hint, error, children }: { label: string; hint?: string; error?: string; children: ReactNode }) {
  return <label className={styles.field}><span>{label}</span>{children}{hint && <small>{hint}</small>}{error && <em>{error}</em>}</label>;
}

function SourceCard({ source }: { source: DataSourceConfig }) {
  return <article className={styles.sourceCard}><div><strong>{source.displayName}</strong><span>{source.provider}</span></div><StatusPill value={source.apiKeyMasked === "未配置" ? "待配置" : "已接入"} /><dl><div><dt>地址</dt><dd>{source.endpoint}</dd></div><div><dt>密钥</dt><dd>{source.apiKeyMasked}</dd></div><div><dt>更新</dt><dd>{stringify(source.metadata)}</dd></div></dl></article>;
}

function StatusSummary({ services }: { services: SettingsPayload["services"] }) {
  const healthy = services.filter((item) => /运行|正常|healthy/i.test(item.status)).length;
  return <section className={styles.panel}><PanelHeading icon={<CheckCircle2 size={17} />} title="运行摘要" description="仅反映服务上报状态，不推断模型可用性。" /><div className={styles.summaryMetric}><strong>{healthy}/{services.length}</strong><span>服务报告正常</span></div><p className={styles.muted}>若真实数据、完整席位、有效模型或盲测证据缺失，预测与回测结论仍保持关闭。</p></section>;
}

function DangerZone({ onComplete }: { onComplete: () => void }) {
  return <section className={`${styles.panel} ${styles.dangerPanel}`}><PanelHeading icon={<AlertTriangle size={17} />} title="危险操作" description="清除本地查询缓存不会删除服务器数据，但会移除本浏览器的受控缓存。" /><Dialog.Root><Dialog.Trigger asChild><button className={styles.dangerButton} type="button"><AlertTriangle size={15} />清除本地查询缓存</button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay className={styles.dialogOverlay} /><Dialog.Content aria-describedby="danger-description" className={styles.dialogContent}><Dialog.Title>确认清除本地缓存？</Dialog.Title><Dialog.Description id="danger-description">该操作会清除当前浏览器的 React Query 缓存并重新读取服务端状态，不会删除服务器数据。</Dialog.Description><div className={styles.dialogActions}><Dialog.Close asChild><button className={styles.outlineButton} type="button">取消</button></Dialog.Close><Dialog.Close asChild><button className={styles.dangerButton} onClick={onComplete} type="button">确认清除</button></Dialog.Close></div></Dialog.Content></Dialog.Portal></Dialog.Root></section>;
}

function ConfigTable({ columns, rows, empty }: { columns: string[]; rows: ReactNode[][]; empty: string }) {
  if (!rows.length) return <EmptyState message={empty} />;
  return <div className={styles.tableWrap}><table className={styles.table}><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td data-label={columns[cellIndex]} key={`${rowIndex}-${cellIndex}`}>{cell}</td>)}</tr>)}</tbody></table></div>;
}

function StatusPill({ value }: { value: string }) { const good = /运行|正常|已启用|已接入|已配置|healthy/i.test(value); const warn = /待|警告|degraded/i.test(value); return <span className={`${styles.statusPill} ${good ? styles.good : warn ? styles.warn : styles.neutral}`}>{value}</span>; }
function ModeBadge({ mode }: { mode?: SettingsPayload["mode"] }) { return <span className={`${styles.modeBadge} ${mode === "java-api" ? styles.modeLive : styles.modeMock}`}>{mode === "java-api" ? "后端业务接口" : "开发模拟"}</span>; }
function NoticeBanner({ notice, onClose }: { notice: Exclude<Notice, null>; onClose: () => void }) { return <div className={`${styles.notice} ${notice.tone === "success" ? styles.noticeSuccess : styles.noticeError}`} role="status"><span>{notice.tone === "success" ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}</span><p>{notice.text}</p><button onClick={onClose} type="button">关闭</button></div>; }
function LoadingState() { return <section className={styles.state}><strong>正在加载系统设置…</strong><span /><span /><span /></section>; }
function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) { return <section className={styles.state} role="alert"><strong>加载系统设置失败</strong><p>{message}</p><button className={styles.primaryButton} onClick={onRetry} type="button">重新加载</button></section>; }
function EmptyState({ message }: { message: string }) { return <div className={styles.emptyState}>{message}</div>; }
function readString(value: unknown, fallback: string) { return typeof value === "string" && value.trim() ? value : fallback; }
function readNumber(value: unknown, fallback: number) { return typeof value === "number" && Number.isFinite(value) ? value : fallback; }
function stringify(value: Record<string, unknown>) { const text = Object.entries(value).map(([key, item]) => `${key}: ${String(item)}`).join("；"); return text || "—"; }
function formatDate(value: string) { const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date); }
function toErrorMessage(error: unknown, fallback: string) { return error instanceof Error && error.message ? error.message : fallback; }
