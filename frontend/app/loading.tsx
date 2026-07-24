import { StateFrame } from "@/components/StateFrame";
import { TopBar } from "@/components/TopBar";
import { loadingState } from "@/lib/api";

export default function Loading() {
  return (
    <>
      <TopBar title="SeatFlow" icon="⌁" />
      <StateFrame state={loadingState()}>{() => null}</StateFrame>
    </>
  );
}
