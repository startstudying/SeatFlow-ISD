import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SeatFlow V3",
  description: "Dragon-Tiger-List capital propagation research console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-gray-50 text-gray-900 antialiased">
        {children}
      </body>
    </html>
  );
}
