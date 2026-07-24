export type SettingsSection =
  | "general"
  | "data-sources"
  | "model"
  | "experiment"
  | "monitoring"
  | "users"
  | "audit";

export type JsonObject = Record<string, unknown>;

export type SettingEntry = {
  key: string;
  category: string;
  value: JsonObject;
  version: number;
};

export type DataSourceConfig = {
  id: string;
  displayName: string;
  provider: string;
  endpoint: string;
  metadata: JsonObject;
  apiKeyMasked: string;
  apiKeyKeyId: string | null;
  version: number;
};

export type ModelConfig = {
  id: string;
  displayName: string;
  provider: string;
  status: string;
  parameters: JsonObject;
  version: number;
};

export type ExperimentConfig = {
  id: string;
  name: string;
  status: string;
  config: JsonObject;
  version: number;
};

export type ServiceMonitor = {
  serviceId: string;
  status: string;
  checkedAt: string;
  metrics: JsonObject;
  version: number;
};

export type UserAccount = {
  id: string;
  displayName: string;
  active: boolean;
  roles: string[];
  version: number;
};

export type RoleDefinition = {
  code: string;
  name: string;
  permissions: string[];
  version: number;
};

export type PermissionDefinition = {
  code: string;
  name: string;
  resource: string;
  version: number;
};

export type AuditLogEntry = {
  id: number;
  actor: string;
  action: string;
  resourceType: string;
  resourceId: string;
  requestId: string;
  idempotencyKey: string | null;
  detail: JsonObject;
  createdAt: string;
};

export type SettingsPayload = {
  mode: "mock" | "java-api";
  settings: SettingEntry[];
  dataSources: DataSourceConfig[];
  models: ModelConfig[];
  experiments: ExperimentConfig[];
  services: ServiceMonitor[];
  users: UserAccount[];
  roles: RoleDefinition[];
  permissions: PermissionDefinition[];
  auditLogs: AuditLogEntry[];
};

export type GeneralSettingsInput = {
  systemName: string;
  timezone: string;
  market: string;
  retentionDays: number;
};

export type DataSourceInput = {
  displayName: string;
  provider: string;
  endpoint: string;
  apiKey?: string;
  metadata: JsonObject;
  expectedVersion: number;
};

export type ModelInput = {
  displayName: string;
  provider: string;
  status: string;
  parameters: JsonObject;
  expectedVersion: number;
};

export type ExperimentInput = {
  name: string;
  status: string;
  config: JsonObject;
  expectedVersion: number;
};

export const settingsSectionLabels: Record<SettingsSection, string> = {
  general: "通用设置",
  "data-sources": "数据源配置",
  model: "模型配置",
  experiment: "实验配置",
  monitoring: "服务监控",
  users: "用户权限",
  audit: "日志审计"
};
