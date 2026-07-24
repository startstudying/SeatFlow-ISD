import { fetchJavaDataCenter } from "./java-api-adapter";
import { fetchMockDataCenter } from "./mock-service";
import type { AssetFilters, DataAsset, DataCenterPayload, PaginatedAssets } from "./types";

export function shouldUseMockDataCenter(): boolean {
  return process.env.NEXT_PUBLIC_SEATFLOW_DATA_CENTER_SOURCE !== "java-api";
}

export async function fetchDataCenter(): Promise<DataCenterPayload> {
  if (shouldUseMockDataCenter()) {
    return fetchMockDataCenter();
  }

  return fetchJavaDataCenter();
}

export function filterAssets(assets: DataAsset[], filters: AssetFilters): PaginatedAssets {
  const keyword = filters.keyword.trim().toLowerCase();
  const filtered = assets.filter((asset) => {
    const keywordMatched =
      !keyword ||
      asset.dataset.toLowerCase().includes(keyword) ||
      asset.provider.toLowerCase().includes(keyword) ||
      asset.owner.toLowerCase().includes(keyword);
    const storeMatched = filters.store === "all" || asset.store === filters.store;
    const statusMatched = filters.status === "all" || asset.status === filters.status;
    return keywordMatched && storeMatched && statusMatched;
  });
  const pageCount = Math.max(1, Math.ceil(filtered.length / filters.pageSize));
  const page = Math.min(Math.max(1, filters.page), pageCount);
  const start = (page - 1) * filters.pageSize;

  return {
    rows: filtered.slice(start, start + filters.pageSize),
    total: filtered.length,
    page,
    pageSize: filters.pageSize
  };
}
