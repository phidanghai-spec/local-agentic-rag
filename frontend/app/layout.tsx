import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Local Agentic RAG & MCP Assistant v2.0",
  description:
    "LangGraph CRAG + Hybrid Search (BM25 + Chroma Cosine + FlashRank) + Semantic Cache + FastMCP ERP + Offline Telemetry",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 selection:bg-sky-500/30 selection:text-sky-200">
        {children}
      </body>
    </html>
  );
}
