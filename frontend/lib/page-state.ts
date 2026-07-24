import { buildStubState, type UiState } from "./api";

type SearchParams = Promise<Record<string, string | string[] | undefined>> | Record<string, string | string[] | undefined>;

export async function maybeStubState<T>(
  searchParams: SearchParams | undefined,
  data: T
): Promise<UiState<T> | null> {
  if (process.env.NEXT_PUBLIC_SEATFLOW_ENABLE_STUBS !== "1") {
    return null;
  }
  const params = await searchParams;
  const raw = Array.isArray(params?.ui_state) ? params?.ui_state[0] : params?.ui_state;
  if (raw === "loading" || raw === "empty" || raw === "error" || raw === "degraded" || raw === "real") {
    return buildStubState(raw, data);
  }
  return null;
}
