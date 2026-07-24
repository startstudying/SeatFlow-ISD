import { backendGet, backendMutation } from "@/lib/backend-client";
import type {
  DataSourceConfig,
  DataSourceInput,
  ExperimentConfig,
  ExperimentInput,
  GeneralSettingsInput,
  ModelConfig,
  ModelInput,
  SettingsPayload,
  SettingEntry
} from "./types";

export async function fetchJavaSettings(): Promise<SettingsPayload> {
  const [settings, dataSources, models, experiments, services, users, roles, permissions, auditLogs] = await Promise.all([
    backendGet<SettingsPayload["settings"]>("/api/v1/settings"),
    backendGet<SettingsPayload["dataSources"]>("/api/v1/data-sources"),
    backendGet<SettingsPayload["models"]>("/api/v1/models"),
    backendGet<SettingsPayload["experiments"]>("/api/v1/experiments"),
    backendGet<SettingsPayload["services"]>("/api/v1/monitoring/services"),
    backendGet<SettingsPayload["users"]>("/api/v1/users"),
    backendGet<SettingsPayload["roles"]>("/api/v1/roles"),
    backendGet<SettingsPayload["permissions"]>("/api/v1/permissions"),
    backendGet<SettingsPayload["auditLogs"]>("/api/v1/audit-logs?limit=100")
  ]);
  return { mode: "java-api", settings, dataSources, models, experiments, services, users, roles, permissions, auditLogs };
}

export function saveJavaGeneral(input: GeneralSettingsInput, version: number): Promise<SettingEntry> {
  return backendMutation("/api/v1/settings/general", "PUT", { category: "general", value: input, expectedVersion: version }, version);
}

export function saveJavaDataSource(id: string, input: DataSourceInput): Promise<DataSourceConfig> {
  return backendMutation(`/api/v1/data-sources/${encodeURIComponent(id)}`, "PUT", input, input.expectedVersion);
}

export function saveJavaModel(id: string, input: ModelInput): Promise<ModelConfig> {
  return backendMutation(`/api/v1/models/${encodeURIComponent(id)}`, "PUT", input, input.expectedVersion);
}

export function saveJavaExperiment(id: string, input: ExperimentInput): Promise<ExperimentConfig> {
  return backendMutation(`/api/v1/experiments/${encodeURIComponent(id)}`, "PUT", input, input.expectedVersion);
}
