import { fetchJavaSettings, saveJavaDataSource, saveJavaExperiment, saveJavaGeneral, saveJavaModel } from "./java-api-adapter";
import { fetchMockSettings } from "./mock-service";
import type { DataSourceConfig, DataSourceInput, ExperimentConfig, ExperimentInput, GeneralSettingsInput, ModelConfig, ModelInput, SettingEntry, SettingsPayload } from "./types";

export function shouldUseMockSettings(): boolean {
  return process.env.NEXT_PUBLIC_SEATFLOW_SETTINGS_SOURCE !== "java-api";
}

export async function fetchSettings(): Promise<SettingsPayload> {
  return shouldUseMockSettings() ? fetchMockSettings() : fetchJavaSettings();
}

export async function saveGeneral(input: GeneralSettingsInput, version: number): Promise<SettingEntry> {
  if (shouldUseMockSettings()) {
    return { key: "general", category: "general", value: input, version: version + 1 };
  }
  return saveJavaGeneral(input, version);
}

export async function saveDataSource(id: string, input: DataSourceInput): Promise<DataSourceConfig> {
  if (shouldUseMockSettings()) {
    return { id, ...input, apiKeyMasked: input.apiKey?.trim() ? "已更新（仅显示状态）" : "已配置（仅显示状态）", apiKeyKeyId: input.apiKey?.trim() ? "mock-key" : null, version: input.expectedVersion + 1 };
  }
  return saveJavaDataSource(id, input);
}

export async function saveModel(id: string, input: ModelInput): Promise<ModelConfig> {
  if (shouldUseMockSettings()) {
    return { id, ...input, version: input.expectedVersion + 1 };
  }
  return saveJavaModel(id, input);
}

export async function saveExperiment(id: string, input: ExperimentInput): Promise<ExperimentConfig> {
  if (shouldUseMockSettings()) {
    return { id, ...input, version: input.expectedVersion + 1 };
  }
  return saveJavaExperiment(id, input);
}
