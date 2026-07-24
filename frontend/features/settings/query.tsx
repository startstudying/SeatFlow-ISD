"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchSettings, saveDataSource, saveExperiment, saveGeneral, saveModel } from "./service";
import type { DataSourceInput, ExperimentInput, GeneralSettingsInput, ModelInput, SettingsPayload } from "./types";

export const settingsQueryKey = ["settings"] as const;

export function useSettingsQuery() {
  return useQuery<SettingsPayload, Error>({ queryKey: settingsQueryKey, queryFn: fetchSettings, staleTime: 30_000, retry: 1 });
}

export function useSettingsMutations() {
  const queryClient = useQueryClient();
  const refresh = async () => { await queryClient.invalidateQueries({ queryKey: settingsQueryKey }); };
  return {
    saveGeneral: useMutation({ mutationFn: ({ input, version }: { input: GeneralSettingsInput; version: number }) => saveGeneral(input, version), onSuccess: refresh }),
    saveDataSource: useMutation({ mutationFn: ({ id, input }: { id: string; input: DataSourceInput }) => saveDataSource(id, input), onSuccess: refresh }),
    saveModel: useMutation({ mutationFn: ({ id, input }: { id: string; input: ModelInput }) => saveModel(id, input), onSuccess: refresh }),
    saveExperiment: useMutation({ mutationFn: ({ id, input }: { id: string; input: ExperimentInput }) => saveExperiment(id, input), onSuccess: refresh })
  };
}
