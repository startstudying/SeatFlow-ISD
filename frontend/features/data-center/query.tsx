"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchDataCenter } from "./service";
import type { DataCenterPayload } from "./types";

export const dataCenterQueryKey = ["data-center"] as const;

export function useDataCenterQuery() {
  return useQuery<DataCenterPayload, Error>({
    queryKey: dataCenterQueryKey,
    queryFn: fetchDataCenter,
    staleTime: 30_000,
    retry: 1
  });
}
