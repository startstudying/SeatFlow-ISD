import type { DataAssetStatus, SyncTaskStatus } from "./types";

export const sectionLabels = {
  overview: "总览",
  assets: "资产",
  quality: "质量",
  graph: "图谱",
  versions: "版本",
  sync: "同步"
} as const;

export const storeLabels = {
  raw: "原始层",
  processed: "处理层",
  features: "特征层",
  graphs: "图谱层"
} as const;

export const assetStatusLabels: Record<DataAssetStatus, string> = {
  ready: "就绪",
  syncing: "同步中",
  degraded: "降级",
  empty: "空"
};

export const taskStatusLabels: Record<SyncTaskStatus, string> = {
  queued: "排队",
  running: "运行中",
  success: "成功",
  failed: "失败"
};

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("zh-CN").format(value);
}

export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}
