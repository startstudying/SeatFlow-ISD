import type { Metadata } from "next";
import { Suspense, type ReactNode } from "react";
import { AppShell } from "@/components/AppShell";
import { QueryProvider } from "@/components/providers/QueryProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "SeatFlow",
  description: "SeatFlow 图谱、预测和实验的证据优先研究界面。"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <QueryProvider>
          <Suspense fallback={<main className="main">{children}</main>}>
            <AppShell>{children}</AppShell>
          </Suspense>
        </QueryProvider>
      </body>
    </html>
  );
}
