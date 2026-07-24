"use client";

import * as echarts from "echarts";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { assetStatusLabels, formatDateTime, formatNumber, formatPercent, sectionLabels, storeLabels, taskStatusLabels } from "@/features/data-center/formatters";
import { filterAssets } from "@/features/data-center/service";
import type { AssetFilters, DataAssetStatus, DataCenterPayload, DataCenterSection } from "@/features/data-center/types";
import { useDataCenterQuery } from "@/features/data-center/query";
import styles from "./DataCenterPage.module.css";

const sections = Object.keys(sectionLabels) as DataCenterSection[];
const storeOptions: AssetFilters["store"][] = ["all", "raw", "processed", "features", "graphs"];
const statusOptions: Array<"all" | DataAssetStatus> = ["all", "ready", "syncing", "degraded", "empty"];
export function DataCenterPage() {
  return <DataCenterContent />;
}

function DataCenterContent() {
  const query = useDataCenterQuery();
  const searchParams = useSearchParams();
  const selectedTab = searchParams.get("tab");
  const activeSection = sections.includes(selectedTab as DataCenterSection) ? selectedTab as DataCenterSection : "overview";
  const [filters, setFilters] = useState<AssetFilters>({
    keyword: "",
    store: "all",
    status: "all",
    page: 1,
    pageSize: 4
  });

  const paginated = useMemo(
    () => filterAssets(query.data?.assets ?? [], filters),
    [filters, query.data?.assets]
  );

  return (
    <main className={styles.page}>
      <PageHeader
        activeSection={activeSection}
        data={query.data}
        isFetching={query.isFetching}
        onRefresh={() => void query.refetch()}
      />

      {query.isLoading && <LoadingState />}
      {query.status === "error" && <ErrorState message={query.error?.message ?? "数据中心加载失败"} onRetry={() => void query.refetch()} />}
      {query.status === "success" && query.data && (
        <>
          {activeSection === "overview" && <section className={styles.sectionPanel} id="overview">
            <SectionTitle title="数据总览" subtitle={`来源：${query.data.mode === "java-api" ? "后端业务接口" : "开发模拟服务"}，交易日 ${query.data.overview.tradeDate}`} />
            <OverviewCards data={query.data} />
          </section>}

          {activeSection === "assets" && <section className={styles.sectionPanel} id="assets">
            <SectionTitle title="数据资产" subtitle="支持关键词、存储层、状态筛选与分页。" />
            <AssetTable filters={filters} pageData={paginated} onFiltersChange={setFilters} />
          </section>}

          {activeSection === "quality" && <section className={styles.sectionPanel} id="quality">
            <SectionTitle title="质量闸门" subtitle="完整性、有效性、时效和阻塞项独立呈现。" />
            <QualityGrid data={query.data} />
          </section>}

          {activeSection === "graph" && <section className={styles.sectionPanel} id="graph">
            <SectionTitle title="关系图谱" subtitle="使用图谱组件渲染；依赖异常时使用内置只读 SVG 降级图。" />
            <GraphPanel data={query.data} />
          </section>}

          {activeSection === "versions" && <section className={styles.sectionPanel} id="versions">
            <SectionTitle title="版本血缘" subtitle="展示数据集版本、内容校验值与生产者。" />
            <Versions data={query.data} />
          </section>}

          {activeSection === "sync" && <section className={styles.sectionPanel} id="sync">
            <SectionTitle title="同步任务" subtitle="队列、运行、成功、失败状态均有可见反馈。" />
            <SyncTasks data={query.data} />
          </section>}
        </>
      )}
    </main>
  );
}

function PageHeader({
  activeSection,
  data,
  isFetching,
  onRefresh
}: {
  activeSection: DataCenterSection;
  data: DataCenterPayload | undefined;
  isFetching: boolean;
  onRefresh: () => void;
}) {
  return (
    <header className={styles.header}>
      <nav className={styles.topNav} aria-label="数据中心顶部导航">
        <Link href="/">SeatFlow</Link>
        <span>数据中心</span>
        <span>{data?.mode === "java-api" ? "后端业务接口" : "开发模拟服务"}</span>
      </nav>
      <div className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>数据中心</p>
          <h1>数据中心</h1>
          <p>统一查看数据资产、质量闸门、图谱快照、版本血缘和同步任务。</p>
        </div>
        <button className={styles.primaryButton} disabled={isFetching} type="button" onClick={onRefresh}>
          {isFetching ? "刷新中..." : "刷新数据"}
        </button>
      </div>
      <nav className={styles.sectionNav} aria-label="数据中心页内导航">
        {sections.map((section) => (
          <Link
            className={activeSection === section ? styles.activeNavItem : undefined}
            href={`/data-center?tab=${section}`}
            key={section}
          >
            {sectionLabels[section]}
          </Link>
        ))}
      </nav>
    </header>
  );
}

function SectionTitle({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className={styles.sectionTitle}>
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function OverviewCards({ data }: { data: DataCenterPayload }) {
  const cards = [
    { label: "资产数", value: formatNumber(data.overview.assetCount), hint: "已注册数据资产" },
    { label: "总行数", value: formatNumber(data.overview.rowCount), hint: "按资产 rows 汇总" },
    { label: "图谱节点", value: formatNumber(data.overview.graphNodeCount), hint: `${formatNumber(data.overview.graphEdgeCount)} 条关系` },
    { label: "质量分", value: formatPercent(data.overview.qualityScore), hint: `${data.overview.blockedIssues} 个阻塞项` },
    { label: "活跃任务", value: formatNumber(data.overview.activeTasks), hint: "排队或运行中" }
  ];

  return (
    <div className={styles.cards}>
      {cards.map((card) => (
        <article className={styles.statusCard} key={card.label}>
          <span>{card.label}</span>
          <strong>{card.value}</strong>
          <p>{card.hint}</p>
        </article>
      ))}
    </div>
  );
}

function AssetTable({
  filters,
  pageData,
  onFiltersChange
}: {
  filters: AssetFilters;
  pageData: ReturnType<typeof filterAssets>;
  onFiltersChange: (filters: AssetFilters) => void;
}) {
  const pageCount = Math.max(1, Math.ceil(pageData.total / pageData.pageSize));

  return (
    <div className={styles.tableCard}>
      <div className={styles.filters}>
        <input
          aria-label="按数据集、数据提供方或责任方搜索"
          placeholder="搜索数据集、数据提供方或责任方"
          type="search"
          value={filters.keyword}
          onChange={(event) => onFiltersChange({ ...filters, keyword: event.target.value, page: 1 })}
        />
        <select
          aria-label="按存储层筛选"
          value={filters.store}
          onChange={(event) => onFiltersChange({ ...filters, store: event.target.value as AssetFilters["store"], page: 1 })}
        >
          {storeOptions.map((store) => (
            <option key={store} value={store}>
              {store === "all" ? "全部存储层" : storeLabels[store]}
            </option>
          ))}
        </select>
        <select
          aria-label="按状态筛选"
          value={filters.status}
          onChange={(event) => onFiltersChange({ ...filters, status: event.target.value as AssetFilters["status"], page: 1 })}
        >
          {statusOptions.map((status) => (
            <option key={status} value={status}>
              {status === "all" ? "全部状态" : assetStatusLabels[status]}
            </option>
          ))}
        </select>
      </div>
      <ResponsiveTable
        columns={["数据集", "存储层", "行数", "版本", "责任方", "状态", "更新"]}
        empty="没有匹配的数据资产。"
        rows={pageData.rows.map((asset) => [
          asset.dataset,
          storeLabels[asset.store],
          formatNumber(asset.rows),
          asset.version,
          asset.owner,
          <Badge tone={asset.status} key={`${asset.id}-status`}>{assetStatusLabels[asset.status]}</Badge>,
          formatDateTime(asset.updatedAt)
        ])}
      />
      <div className={styles.pagination}>
        <span>共 {pageData.total} 条，{pageData.page} / {pageCount} 页</span>
        <div>
          <button disabled={pageData.page <= 1} type="button" onClick={() => onFiltersChange({ ...filters, page: pageData.page - 1 })}>
            上一页
          </button>
          <button disabled={pageData.page >= pageCount} type="button" onClick={() => onFiltersChange({ ...filters, page: pageData.page + 1 })}>
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}

function QualityGrid({ data }: { data: DataCenterPayload }) {
  if (!data.quality.length) {
    return <EmptyState message="暂无质量报告。" />;
  }

  return (
    <div className={styles.qualityGrid}>
      {data.quality.map((item) => (
        <article className={styles.qualityCard} key={item.id}>
          <div className={styles.cardHeader}>
            <div>
              <strong>{item.dataset}</strong>
              <span>{item.provider}</span>
            </div>
            <Badge tone={item.gate === "pass" ? "ready" : item.gate === "warn" ? "degraded" : "empty"}>{item.gate}</Badge>
          </div>
          <Meter label="完整性" value={item.completeness} />
          <Meter label="有效性" value={item.validity} />
          <p className={styles.smallText}>时效 {item.freshnessHours}h，问题 {item.issueCount} 个，检查于 {formatDateTime(item.checkedAt)}</p>
        </article>
      ))}
    </div>
  );
}

function GraphPanel({ data }: { data: DataCenterPayload }) {
  return (
    <div className={styles.graphLayout}>
      <EChartsGraph data={data} />
      <div className={styles.graphAside}>
        <strong>图谱摘要</strong>
        <span>节点：{data.graph.nodes.length}</span>
        <span>关系：{data.graph.edges.length}</span>
        <span>交易日：{data.overview.tradeDate}</span>
        <p>图谱组件会自动渲染力导向图；依赖加载异常时保留 SVG 降级态，避免页面空白。</p>
      </div>
    </div>
  );
}

function EChartsGraph({ data }: { data: DataCenterPayload }) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    let disposed = false;
    let chart: echarts.ECharts | null = null;

    function renderChart() {
      if (!chartRef.current) {
        return;
      }
      try {
        if (disposed || !chartRef.current) {
          return;
        }
        chart = echarts.init(chartRef.current);
        chart.setOption({
          tooltip: {},
          series: [
            {
              type: "graph",
              layout: "force",
              roam: true,
              force: { repulsion: 120, edgeLength: 80 },
              label: { show: true },
              data: data.graph.nodes.map((node) => ({
                id: node.id,
                name: node.name,
                value: node.value,
                category: node.type
              })),
              links: data.graph.edges.map((edge) => ({
                source: edge.source,
                target: edge.target,
                value: edge.weight,
                label: { show: true, formatter: edge.relation }
              })),
              categories: ["seat", "stock", "industry", "concept", "event"].map((name) => ({ name }))
            }
          ]
        });
      } catch {
        setFallback(true);
      }
    }

    renderChart();
    return () => {
      disposed = true;
      chart?.dispose();
    };
  }, [data]);

  if (!data.graph.nodes.length) {
    return <EmptyState message="暂无图谱节点。" />;
  }

  return (
    <div className={styles.chartShell}>
      <div ref={chartRef} className={styles.echartsCanvas} aria-label="ECharts 图谱画布" />
      {fallback && <FallbackGraph data={data} />}
    </div>
  );
}

function FallbackGraph({ data }: { data: DataCenterPayload }) {
  const points = data.graph.nodes.map((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, data.graph.nodes.length);
    return {
      ...node,
      x: 50 + Math.cos(angle) * 32,
      y: 50 + Math.sin(angle) * 32
    };
  });

  return (
    <svg className={styles.fallbackGraph} role="img" aria-label="图谱降级渲染" viewBox="0 0 100 100">
      {data.graph.edges.map((edge) => {
        const source = points.find((point) => point.id === edge.source);
        const target = points.find((point) => point.id === edge.target);
        if (!source || !target) {
          return null;
        }
        return <line key={`${edge.source}-${edge.target}`} x1={source.x} x2={target.x} y1={source.y} y2={target.y} />;
      })}
      {points.map((node) => (
        <g key={node.id}>
          <circle cx={node.x} cy={node.y} r="6" />
          <text x={node.x} y={node.y + 11}>{node.name}</text>
        </g>
      ))}
    </svg>
  );
}

function Versions({ data }: { data: DataCenterPayload }) {
  return (
    <ResponsiveTable
      columns={["数据集", "版本", "标签", "行数", "内容校验值", "创建者", "创建时间"]}
      empty="暂无版本记录。"
      rows={data.versions.map((version) => [
        version.dataset,
        version.version,
        version.tag,
        formatNumber(version.rows),
        version.checksum,
        version.author,
        formatDateTime(version.createdAt)
      ])}
    />
  );
}

function SyncTasks({ data }: { data: DataCenterPayload }) {
  if (!data.syncTasks.length) {
    return <EmptyState message="暂无同步任务。" />;
  }

  return (
    <div className={styles.taskList}>
      {data.syncTasks.map((task) => (
        <article className={styles.taskCard} key={task.id}>
          <div className={styles.cardHeader}>
            <div>
              <strong>{task.target}</strong>
              <span>{task.source}</span>
            </div>
            <Badge tone={task.status === "success" ? "ready" : task.status === "failed" ? "empty" : "syncing"}>{taskStatusLabels[task.status]}</Badge>
          </div>
          <Meter label={task.message} value={task.progress} />
          <p className={styles.smallText}>开始：{formatDateTime(task.startedAt)}</p>
        </article>
      ))}
    </div>
  );
}

function ResponsiveTable({
  columns,
  rows,
  empty
}: {
  columns: string[];
  rows: Array<Array<ReactNode>>;
  empty: string;
}) {
  if (!rows.length) {
    return <EmptyState message={empty} />;
  }

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td data-label={columns[cellIndex]} key={`${rowIndex}-${cellIndex}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className={styles.meter}>
      <div>
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <span className={styles.track}>
        <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </span>
    </div>
  );
}

function Badge({ tone, children }: { tone: DataAssetStatus; children: ReactNode }) {
  return <span className={`${styles.badge} ${styles[tone]}`}>{children}</span>;
}

function LoadingState() {
  return (
    <section className={styles.stateBlock} aria-live="polite">
      <strong>正在加载数据中心...</strong>
      <span />
      <span />
      <span />
    </section>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className={styles.stateBlock} role="alert">
      <strong>加载失败</strong>
      <p>{message}</p>
      <button type="button" onClick={onRetry}>重试</button>
    </section>
  );
}

function EmptyState({ message }: { message: string }) {
  return <div className={styles.emptyState}>{message}</div>;
}
