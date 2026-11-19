"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ApprovalRequiredPayload,
  CacheStats,
  Citation,
  IndexedDocument,
  StreamMetadata,
  TelemetryMetrics,
  TraceRecord,
  clearSemanticCache,
  fetchDocuments,
  fetchHealth,
  fetchMetrics,
  fetchTraces,
  streamAgentChat,
  uploadDocument,
} from "@/lib/agentClient";
import { ErpApprovalCard } from "@/components/ErpApprovalCard";

type ActiveTab = "chat" | "documents" | "telemetry";

interface ThinkingStep {
  node: string;
  message: string;
  timestamp: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps: ThinkingStep[];
  stepsExpanded: boolean;
  metadata?: StreamMetadata;
  approval?: ApprovalRequiredPayload | null;
  approvalResolution?: "approved" | "rejected" | null;
  error?: string | null;
  isStreaming?: boolean;
}

const NODE_ICONS: Record<string, string> = {
  pii_proxy: "🔒",
  semantic_cache: "⚡",
  route_question: "🚦",
  retrieve: "🔍",
  grade_documents: "⚖️",
  rewrite_query: "🔄",
  call_mcp_tools: "🛠️",
  commit_erp_order: "💾",
  generate: "✍️",
  check_hallucination: "🛡️",
};

const SAMPLE_PROMPTS = [
  {
    label: "🔍 Hybrid RAG: Kiến trúc Reranker",
    badge: "Vector + BM25",
    question:
      "Hệ thống sử dụng mô hình Reranker nào và chạy trên phần cứng nào?",
  },
  {
    label: "⚖️ So sánh: X1 vs Pro Z",
    badge: "Query Decomposition",
    question:
      "So sánh thời lượng pin và trọng lượng giữa AI Workstation X1 và EdgeBook Pro Z",
  },
  {
    label: "🛠️ MCP ERP: Tồn kho & Báo giá",
    badge: "FastMCP Read",
    question:
      "Kiểm tra tồn kho của mã AI-WS-X1 và tính tổng tiền nếu mua 2 chiếc với chiết khấu 10%",
  },
  {
    label: "⏸️ HITL ERP: Chốt đơn trừ kho",
    badge: "Approval Gate",
    question: "Chốt đơn đặt mua 2 chiếc AI-WS-X1 với chiết khấu 10%",
  },
  {
    label: "🔒 PII Guard: Ẩn danh SĐT & Email",
    badge: "Presidio Proxy",
    question:
      "Tôi là khách hàng SĐT 0912345678, email test@company.vn muốn hỏi thời gian bảo hành của AI Workstation X1?",
  },
];

function formatInlineMarkdown(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={idx} className="font-semibold text-sky-200">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code
          key={idx}
          className="rounded bg-slate-800/90 px-1.5 py-0.5 font-mono text-xs text-amber-300"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function FormattedAnswer({ content }: { content: string }) {
  if (!content) return null;
  const lines = content.split("\n");
  return (
    <div className="space-y-1.5 text-sm leading-relaxed text-slate-100">
      {lines.map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <div key={i} className="h-1.5" />;
        }
        if (trimmed.startsWith("### ")) {
          return (
            <h4 key={i} className="pt-1 text-sm font-bold text-sky-300">
              {formatInlineMarkdown(trimmed.slice(4))}
            </h4>
          );
        }
        if (trimmed.startsWith("## ")) {
          return (
            <h3 key={i} className="pt-1 text-base font-bold text-sky-300">
              {formatInlineMarkdown(trimmed.slice(3))}
            </h3>
          );
        }
        if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
          return (
            <div key={i} className="flex items-start gap-2 pl-2">
              <span className="text-sky-400 select-none">•</span>
              <span>{formatInlineMarkdown(trimmed.slice(2))}</span>
            </div>
          );
        }
        return <p key={i}>{formatInlineMarkdown(line)}</p>;
      })}
    </div>
  );
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("chat");

  // Health & Cache state
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [cacheBanner, setCacheBanner] = useState<string | null>(null);
  const [clearingCache, setClearingCache] = useState<boolean>(false);

  // Chat tab state
  const [threadId, setThreadId] = useState<string>("session_001");
  const [bypassCache, setBypassCache] = useState<boolean>(false);
  const [questionInput, setQuestionInput] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);

  // Documents tab state
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState<boolean>(false);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [docDomain, setDocDomain] = useState<string>("");
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Telemetry tab state
  const [metrics, setMetrics] = useState<TelemetryMetrics | null>(null);
  const [traces, setTraces] = useState<TraceRecord[]>([]);
  const [telemetryLoading, setTelemetryLoading] = useState<boolean>(false);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);

  useEffect(() => {
    const randomSuffix = Math.random().toString(36).substring(2, 8);
    setThreadId(`session_${randomSuffix}`);
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const res = await fetchHealth();
      setBackendOnline(res.status === "ok");
      if (res.cache_stats) {
        setCacheStats(res.cache_stats);
      }
    } catch {
      setBackendOnline(false);
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    setDocsError(null);
    try {
      const res = await fetchDocuments();
      setDocuments(res.documents ?? []);
    } catch (err) {
      setDocsError(
        err instanceof Error ? err.message : "Không thể tải danh sách tài liệu."
      );
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const loadTelemetry = useCallback(async () => {
    setTelemetryLoading(true);
    setTelemetryError(null);
    try {
      const [m, t] = await Promise.all([fetchMetrics(), fetchTraces(25)]);
      setMetrics(m);
      setTraces(t.traces ?? []);
    } catch (err) {
      setTelemetryError(
        err instanceof Error ? err.message : "Không thể tải dữ liệu Telemetry."
      );
    } finally {
      setTelemetryLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  useEffect(() => {
    if (activeTab === "documents") {
      loadDocuments();
    } else if (activeTab === "telemetry") {
      loadTelemetry();
    }
  }, [activeTab, loadDocuments, loadTelemetry]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const updateAssistantMessage = useCallback(
    (msgId: string, updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === msgId ? updater(m) : m))
      );
    },
    []
  );

  const handleSendQuestion = async (e?: React.FormEvent, customQ?: string) => {
    if (e) e.preventDefault();
    const q = (customQ ?? questionInput).trim();
    if (!q || isStreaming) return;

    if (!customQ) {
      setQuestionInput("");
    }

    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `bot_${Date.now() + 1}`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: q,
      steps: [],
      stepsExpanded: false,
    };

    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      steps: [],
      stepsExpanded: true,
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    await streamAgentChat(
      q,
      threadId,
      {
        onStatus: (node, message) => {
          const step: ThinkingStep = {
            node,
            message,
            timestamp: new Date().toLocaleTimeString("vi-VN"),
          };
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            steps: [...m.steps, step],
          }));
        },
        onToken: (chunk) => {
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            content: m.content + chunk,
          }));
        },
        onApprovalRequired: (payload) => {
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            approval: payload,
            isStreaming: false,
          }));
        },
        onMetadata: (meta) => {
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            metadata: meta,
          }));
        },
        onDone: () => {
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            isStreaming: false,
          }));
          setIsStreaming(false);
          refreshHealth();
        },
        onError: (err) => {
          updateAssistantMessage(assistantMsgId, (m) => ({
            ...m,
            error: err.message,
            isStreaming: false,
          }));
          setIsStreaming(false);
        },
      },
      { bypassCache }
    );
  };

  const handleClearCache = async () => {
    setClearingCache(true);
    setCacheBanner(null);
    try {
      const res = await clearSemanticCache();
      setCacheBanner(`Đã xóa Semantic Cache (${res.cleared} mục).`);
      await refreshHealth();
    } catch (err) {
      setCacheBanner(
        `Lỗi xóa cache: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setClearingCache(false);
    }
  };

  const handleNewSession = () => {
    const suffix = Math.random().toString(36).substring(2, 8);
    setThreadId(`session_${suffix}`);
    setMessages([]);
  };

  const handleUploadDocument = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadStatus(null);
    setDocsError(null);
    try {
      const res = await uploadDocument(selectedFile, docDomain);
      setUploadStatus(
        `✅ Tệp "${res.filename}" (${res.ingest_status ?? res.status}) — ${res.chunk_count} chunks [domain: ${res.doc_domain ?? "auto"}]`
      );
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      await loadDocuments();
    } catch (err) {
      setDocsError(
        err instanceof Error ? err.message : "Lỗi tải lên tài liệu."
      );
    } finally {
      setUploading(false);
    }
  };

  const hasPendingApproval = messages.some(
    (m) => m.approval && !m.approvalResolution
  );

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Navigation Header */}
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 py-3.5">
        <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-lg font-bold tracking-tight text-sky-400">
                Local Agentic RAG &amp; MCP Assistant
              </h1>
              <span className="rounded-md bg-sky-500/10 border border-sky-500/30 px-2 py-0.5 text-[11px] font-semibold text-sky-300">
                v2.0 Production
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              LangGraph CRAG • Hybrid Search (BM25 + Chroma + FlashRank) •
              Semantic Cache • FastMCP HITL • SQLite Telemetry
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Backend Status Badge */}
            <button
              type="button"
              onClick={refreshHealth}
              title="Nhấn để kiểm tra lại kết nối Backend"
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition ${
                backendOnline === true
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  : backendOnline === false
                    ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
                    : "border-slate-700 bg-slate-800 text-slate-400"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline === true
                    ? "bg-emerald-400 animate-pulse"
                    : backendOnline === false
                      ? "bg-rose-400"
                      : "bg-slate-400"
                }`}
              />
              {backendOnline === true
                ? `Backend Online${
                    cacheStats?.entry_count !== undefined
                      ? ` • Cache: ${cacheStats.entry_count}`
                      : ""
                  }`
                : backendOnline === false
                  ? "Backend Offline (:8000)"
                  : "Đang kiểm tra..."}
            </button>

            {/* 3-Tab Switcher */}
            <nav className="inline-flex rounded-xl bg-slate-950 p-1 border border-slate-800">
              <button
                type="button"
                onClick={() => setActiveTab("chat")}
                className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                  activeTab === "chat"
                    ? "bg-sky-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                💬 Agent Chat
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("documents")}
                className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                  activeTab === "documents"
                    ? "bg-sky-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                📚 Documents
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("telemetry")}
                className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                  activeTab === "telemetry"
                    ? "bg-sky-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                📊 Telemetry
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 flex flex-col">
        {/* ============================================================ */}
        {/* TAB 1: AGENT CHAT                                            */}
        {/* ============================================================ */}
        {activeTab === "chat" && (
          <div className="flex-1 flex flex-col gap-4">
            {/* Session & Cache Control Bar */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-3.5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Thread ID:</span>
                  <code className="rounded bg-slate-950 border border-slate-800 px-2 py-0.5 font-mono text-sky-300">
                    {threadId}
                  </code>
                </div>
                <button
                  type="button"
                  onClick={handleNewSession}
                  disabled={isStreaming}
                  className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-2.5 py-1 text-xs text-slate-200 transition disabled:opacity-50"
                >
                  + Phiên mới
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {/* Bypass Cache Toggle */}
                <label className="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={bypassCache}
                    onChange={(e) => setBypassCache(e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-slate-700 bg-slate-950 text-sky-500 focus:ring-0"
                  />
                  <span>Bypass Semantic Cache</span>
                </label>

                {/* Clear Cache Button */}
                <button
                  type="button"
                  onClick={handleClearCache}
                  disabled={clearingCache}
                  className="rounded-lg border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 px-3 py-1 text-xs font-medium text-amber-300 transition disabled:opacity-50"
                >
                  {clearingCache ? "Đang xóa..." : "🗑️ Clear Cache"}
                </button>
              </div>
            </section>

            {cacheBanner && (
              <div className="rounded-xl border border-sky-500/30 bg-sky-950/30 px-4 py-2 text-xs text-sky-200 flex items-center justify-between">
                <span>{cacheBanner}</span>
                <button
                  type="button"
                  onClick={() => setCacheBanner(null)}
                  className="text-slate-400 hover:text-slate-200"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Quick Sample Questions */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
                Kịch bản kiểm thử nhanh (Click để gửi ngay)
              </p>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_PROMPTS.map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    disabled={isStreaming || hasPendingApproval}
                    onClick={() => handleSendQuestion(undefined, sample.question)}
                    className="group text-left rounded-lg border border-slate-800 bg-slate-950/80 hover:border-sky-500/50 hover:bg-slate-900 px-3 py-2 text-xs transition disabled:opacity-50"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-200 group-hover:text-sky-300">
                        {sample.label}
                      </span>
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
                        {sample.badge}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </section>

            {/* Chat Messages Feed */}
            <section className="flex-1 space-y-4 overflow-y-auto min-h-[380px] rounded-xl border border-slate-800/80 bg-slate-900/30 p-4">
              {messages.length === 0 ? (
                <div className="h-full min-h-[300px] flex flex-col items-center justify-center text-center text-slate-500 space-y-2">
                  <div className="text-3xl">🤖</div>
                  <p className="text-sm font-medium text-slate-300">
                    Sẵn sàng truy vấn Local Agentic RAG &amp; FastMCP ERP
                  </p>
                  <p className="text-xs max-w-md text-slate-500">
                    Chọn một kịch bản mẫu phía trên hoặc nhập câu hỏi bên dưới.
                    Tiến trình suy nghĩ của các Node (`node_status`), trích dẫn
                    FlashRank và thẻ phê duyệt đơn hàng HITL sẽ hiển thị trực
                    tiếp theo thời gian thực.
                  </p>
                </div>
              ) : (
                messages.map((msg) => {
                  if (msg.role === "user") {
                    return (
                      <div key={msg.id} className="flex justify-end">
                        <div className="max-w-2xl rounded-2xl rounded-tr-sm bg-sky-600 px-4 py-2.5 text-sm text-white shadow">
                          {msg.content}
                        </div>
                      </div>
                    );
                  }

                  const meta = msg.metadata;
                  const citations: Citation[] = meta?.citations ?? [];

                  return (
                    <div
                      key={msg.id}
                      className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 space-y-3 shadow"
                    >
                      {/* Collapsible Thinking Steps (node_status) */}
                      {msg.steps.length > 0 && (
                        <div className="rounded-lg border border-slate-800 bg-slate-950/70 overflow-hidden">
                          <button
                            type="button"
                            onClick={() =>
                              updateAssistantMessage(msg.id, (m) => ({
                                ...m,
                                stepsExpanded: !m.stepsExpanded,
                              }))
                            }
                            className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-900/60 transition"
                          >
                            <span className="flex items-center gap-2">
                              <span>
                                {msg.isStreaming ? "⏳" : "🧠"} Thinking Steps (
                                {msg.steps.length} bước)
                              </span>
                              <span className="text-[11px] font-normal text-slate-400">
                                {msg.steps[msg.steps.length - 1]?.node}
                              </span>
                            </span>
                            <span className="text-slate-400">
                              {msg.stepsExpanded ? "▲ Thu gọn" : "▼ Mở rộng"}
                            </span>
                          </button>

                          {msg.stepsExpanded && (
                            <ul className="border-t border-slate-800/80 px-3 py-2 space-y-1.5 text-xs text-slate-300">
                              {msg.steps.map((s, i) => (
                                <li
                                  key={i}
                                  className="flex items-start justify-between gap-2"
                                >
                                  <div className="flex items-start gap-2">
                                    <span>{NODE_ICONS[s.node] ?? "⚡"}</span>
                                    <span className="font-mono text-[11px] text-sky-400">
                                      [{s.node}]
                                    </span>
                                    <span>{s.message}</span>
                                  </div>
                                  <span className="text-[10px] text-slate-500 shrink-0">
                                    {s.timestamp}
                                  </span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}

                      {/* Metadata Badges (Cache, PII, Blocked, Grounded, HITL, Latency) */}
                      {meta && (
                        <div className="flex flex-wrap items-center gap-2">
                          {meta.cache_hit && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/40 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
                              ⚡ Cache HIT
                              {meta.similarity !== undefined &&
                              meta.similarity !== null
                                ? ` (${(meta.similarity * 100).toFixed(1)}%)`
                                : ""}
                            </span>
                          )}

                          {meta.cache_hit === false && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 border border-slate-700 px-2.5 py-0.5 text-[11px] text-slate-400">
                              Cache MISS
                            </span>
                          )}

                          {meta.pii_redacted && meta.pii_redacted.length > 0 && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/40 px-2.5 py-0.5 text-[11px] font-semibold text-amber-300">
                              🔒 PII Masked: {meta.pii_redacted.join(", ")}
                            </span>
                          )}

                          {meta.blocked && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/20 border border-rose-500/40 px-2.5 py-0.5 text-[11px] font-semibold text-rose-300">
                              🛡️ Prompt Injection Blocked
                            </span>
                          )}

                          {meta.hallucination_grade && (
                            <span
                              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${
                                meta.hallucination_grade === "yes"
                                  ? "border-sky-500/30 bg-sky-500/10 text-sky-300"
                                  : "border-rose-500/30 bg-rose-500/10 text-rose-300"
                              }`}
                            >
                              Self-RAG Grounded: {meta.hallucination_grade}
                            </span>
                          )}

                          {meta.approval_status &&
                            meta.approval_status !== "none" && (
                              <span
                                className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${
                                  meta.approval_status === "approved"
                                    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                                    : "border-rose-500/40 bg-rose-500/15 text-rose-300"
                                }`}
                              >
                                ERP Order: {meta.approval_status.toUpperCase()}
                              </span>
                            )}

                          {meta.latency_ms !== undefined && (
                            <span className="inline-flex items-center rounded-full bg-slate-800/90 border border-slate-700 px-2.5 py-0.5 text-[11px] font-mono text-slate-300">
                              ⏱️ {meta.latency_ms} ms
                            </span>
                          )}
                        </div>
                      )}

                      {/* Streaming Answer Content */}
                      {msg.content ? (
                        <FormattedAnswer content={msg.content} />
                      ) : msg.isStreaming ? (
                        <p className="text-xs text-slate-400 italic animate-pulse">
                          Đang nhận luồng dữ liệu từ Agent...
                        </p>
                      ) : null}

                      {/* Human-in-the-Loop ERP Approval Card */}
                      {msg.approval && !msg.approvalResolution && (
                        <ErpApprovalCard
                          approval={msg.approval}
                          handlers={{
                            onStatus: (node, message) => {
                              updateAssistantMessage(msg.id, (m) => ({
                                ...m,
                                steps: [
                                  ...m.steps,
                                  {
                                    node,
                                    message,
                                    timestamp: new Date().toLocaleTimeString(
                                      "vi-VN"
                                    ),
                                  },
                                ],
                              }));
                            },
                            onToken: (chunk) => {
                              updateAssistantMessage(msg.id, (m) => ({
                                ...m,
                                content: m.content + chunk,
                              }));
                            },
                            onMetadata: (newMeta) => {
                              updateAssistantMessage(msg.id, (m) => ({
                                ...m,
                                metadata: { ...m.metadata, ...newMeta },
                              }));
                            },
                          }}
                          onResolved={(decision) => {
                            updateAssistantMessage(msg.id, (m) => ({
                              ...m,
                              approvalResolution: decision,
                            }));
                            refreshHealth();
                          }}
                        />
                      )}

                      {/* Error Display */}
                      {msg.error && (
                        <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 px-3 py-2 text-xs text-rose-300">
                          ⚠️ Lỗi: {msg.error}
                        </div>
                      )}

                      {/* Citations List */}
                      {citations.length > 0 && (
                        <div className="pt-2 border-t border-slate-800 space-y-2">
                          <div className="text-xs font-semibold text-slate-400">
                            📑 Nguồn trích dẫn ({citations.length}):
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {citations.map((c, idx) => {
                              const score = c.relevance_score ?? c.score;
                              return (
                                <div
                                  key={idx}
                                  className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5 text-xs space-y-1"
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="font-semibold text-sky-400 truncate">
                                      {c.source}
                                    </span>
                                    {score !== undefined && (
                                      <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                                        FlashRank: {score}
                                      </span>
                                    )}
                                  </div>
                                  {c.snippet && (
                                    <p className="text-slate-400 line-clamp-2 text-[11px]">
                                      {c.snippet}
                                    </p>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
              <div ref={chatBottomRef} />
            </section>

            {/* Chat Input Bar */}
            <form
              onSubmit={(e) => handleSendQuestion(e)}
              className="rounded-xl border border-slate-800 bg-slate-900 p-3 flex gap-3 sticky bottom-4 shadow-xl"
            >
              <input
                type="text"
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                disabled={isStreaming || hasPendingApproval}
                placeholder={
                  hasPendingApproval
                    ? "Vui lòng Phê duyệt hoặc Từ chối đơn hàng ERP phía trên để tiếp tục..."
                    : "Nhập câu hỏi cho Local Agentic RAG (hỗ trợ tiếng Việt, mã SKU, so sánh sản phẩm)..."
                }
                className="flex-1 rounded-lg border border-slate-800 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={
                  isStreaming || hasPendingApproval || !questionInput.trim()
                }
                className="rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 px-5 py-2.5 text-sm font-semibold text-white transition"
              >
                {isStreaming ? "Đang xử lý..." : "Gửi câu hỏi"}
              </button>
            </form>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 2: DOCUMENTS (MULTI-FORMAT INGESTION & MANIFEST)         */}
        {/* ============================================================ */}
        {activeTab === "documents" && (
          <div className="space-y-6">
            {/* Upload Document Card */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 space-y-4">
              <div>
                <h2 className="text-base font-bold text-sky-400">
                  📤 Tải lên &amp; Lập chỉ mục Tài liệu mới (Incremental SHA-256)
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Hỗ trợ tệp <code>.md</code>, <code>.txt</code>,{" "}
                  <code>.pdf</code>. Hệ thống tự động kiểm tra mã băm SHA-256
                  để tránh nhúng trùng lặp vào ChromaDB.
                </p>
              </div>

              <form
                onSubmit={handleUploadDocument}
                className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end"
              >
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">
                    Chọn tệp tài liệu (.md, .txt, .pdf)
                  </label>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".md,.txt,.pdf"
                    onChange={(e) =>
                      setSelectedFile(e.target.files?.[0] ?? null)
                    }
                    className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-sky-600 file:px-3 file:py-1 file:text-xs file:font-semibold file:text-white hover:file:bg-sky-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">
                    Nhãn miền tài liệu (<code>doc_domain</code> - tùy chọn)
                  </label>
                  <input
                    type="text"
                    list="domain-presets"
                    value={docDomain}
                    onChange={(e) => setDocDomain(e.target.value)}
                    placeholder="VD: product_catalog, system_architecture"
                    className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-sky-500 focus:outline-none"
                  />
                  <datalist id="domain-presets">
                    <option value="product_catalog" />
                    <option value="system_architecture" />
                    <option value="company_policy" />
                  </datalist>
                </div>

                <div>
                  <button
                    type="submit"
                    disabled={!selectedFile || uploading}
                    className="w-full rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 px-4 py-2.5 text-xs font-semibold text-white transition"
                  >
                    {uploading
                      ? "Đang băm SHA-256 & lập chỉ mục..."
                      : "⚡ Tải lên & Lập chỉ mục"}
                  </button>
                </div>
              </form>

              {uploadStatus && (
                <div className="rounded-lg border border-emerald-500/40 bg-emerald-950/30 px-3.5 py-2.5 text-xs text-emerald-300">
                  {uploadStatus}
                </div>
              )}

              {docsError && (
                <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 px-3.5 py-2.5 text-xs text-rose-300">
                  ⚠️ {docsError}
                </div>
              )}
            </section>

            {/* Indexed Documents Table */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-200">
                    📚 Danh sách Tài liệu đã Lập chỉ mục ({documents.length})
                  </h3>
                  <p className="text-xs text-slate-400">
                    Trích xuất trực tiếp từ <code>data/ingestion_manifest.json</code>
                  </p>
                </div>
                <button
                  type="button"
                  onClick={loadDocuments}
                  disabled={docsLoading}
                  className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 transition disabled:opacity-50"
                >
                  {docsLoading ? "Đang tải..." : "🔄 Làm mới"}
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2.5 px-3 font-semibold">Tên tệp</th>
                      <th className="py-2.5 px-3 font-semibold">
                        Miền (doc_domain)
                      </th>
                      <th className="py-2.5 px-3 font-semibold">Số Chunks</th>
                      <th className="py-2.5 px-3 font-semibold">
                        SHA-256 Fingerprint
                      </th>
                      <th className="py-2.5 px-3 font-semibold">
                        Cập nhật lúc
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {documents.length === 0 ? (
                      <tr>
                        <td
                          colSpan={5}
                          className="py-8 text-center text-slate-500"
                        >
                          {docsLoading
                            ? "Đang tải danh sách tài liệu..."
                            : "Chưa có tài liệu nào trong manifest. Hãy tải lên tệp đầu tiên!"}
                        </td>
                      </tr>
                    ) : (
                      documents.map((doc) => (
                        <tr
                          key={doc.filename}
                          className="hover:bg-slate-800/40 transition"
                        >
                          <td className="py-2.5 px-3 font-medium text-sky-300">
                            {doc.filename}
                          </td>
                          <td className="py-2.5 px-3">
                            <span className="rounded-full bg-slate-800 border border-slate-700 px-2.5 py-0.5 text-[11px] text-slate-300">
                              {doc.doc_domain || "default"}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-emerald-300">
                            {doc.chunk_count}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-400">
                            {doc.sha256 ? `${doc.sha256.slice(0, 16)}...` : "-"}
                          </td>
                          <td className="py-2.5 px-3 text-slate-400">
                            {doc.updated_at
                              ? new Date(doc.updated_at).toLocaleString("vi-VN")
                              : "-"}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 3: TELEMETRY (OFFLINE MINI-LANGSMITH OBSERVABILITY)      */}
        {/* ============================================================ */}
        {activeTab === "telemetry" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-sky-400">
                  📊 Offline Telemetry &amp; Trace Explorer
                </h2>
                <p className="text-xs text-slate-400">
                  Giám sát độ trễ p50/p95, tỉ lệ trúng Semantic Cache, thống kê
                  HITL và lịch sử từng Node trong SQLite cục bộ.
                </p>
              </div>
              <button
                type="button"
                onClick={loadTelemetry}
                disabled={telemetryLoading}
                className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3.5 py-1.5 text-xs font-medium text-slate-200 transition disabled:opacity-50"
              >
                {telemetryLoading ? "Đang đồng bộ..." : "🔄 Làm mới số liệu"}
              </button>
            </div>

            {telemetryError && (
              <div className="rounded-xl border border-rose-500/40 bg-rose-950/30 px-4 py-3 text-xs text-rose-300">
                ⚠️ {telemetryError}
              </div>
            )}

            {/* KPI Cards */}
            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
                <p className="text-xs text-slate-400">Tổng số lượt truy vấn</p>
                <p className="mt-1 text-2xl font-bold text-slate-100">
                  {metrics?.total_requests ?? 0}
                </p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Ghi nhận trong <code>data/telemetry.sqlite</code>
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
                <p className="text-xs text-slate-400">
                  Semantic Cache Hit Rate
                </p>
                <p className="mt-1 text-2xl font-bold text-emerald-400">
                  {((metrics?.cache_hit_rate ?? 0) * 100).toFixed(1)}%
                </p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Cosine threshold ≥ 0.92
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
                <p className="text-xs text-slate-400">
                  Độ trễ p50 / p95 (Latency)
                </p>
                <p className="mt-1 text-2xl font-bold text-sky-400 font-mono">
                  {metrics?.p50_latency_ms ?? 0}{" "}
                  <span className="text-sm font-normal text-slate-400">
                    / {metrics?.p95_latency_ms ?? 0} ms
                  </span>
                </p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Phân vị nội suy tuyến tính
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4">
                <p className="text-xs text-slate-400">
                  Thống kê HITL &amp; Bảo mật
                </p>
                <div className="mt-2 flex flex-wrap gap-1.5 text-xs">
                  {metrics &&
                  Object.keys(metrics.hitl_counts || {}).length > 0 ? (
                    Object.entries(metrics.hitl_counts).map(
                      ([status, count]) => (
                        <span
                          key={status}
                          className="rounded-md bg-slate-950 border border-slate-800 px-2 py-0.5 text-[11px]"
                        >
                          <span className="text-slate-400">{status}:</span>{" "}
                          <strong className="text-amber-300">{count}</strong>
                        </span>
                      )
                    )
                  ) : (
                    <span className="text-slate-500 text-xs">
                      Chưa có sự kiện
                    </span>
                  )}
                </div>
              </div>
            </section>

            {/* Recent Traces Table */}
            <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 space-y-4">
              <h3 className="text-sm font-bold text-slate-200">
                🕵️ Các lượt chạy gần nhất (Recent Execution Traces)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2.5 px-3 font-semibold">Thời gian</th>
                      <th className="py-2.5 px-3 font-semibold">Thread</th>
                      <th className="py-2.5 px-3 font-semibold">
                        Câu hỏi (Đã ẩn danh PII)
                      </th>
                      <th className="py-2.5 px-3 font-semibold">Datasource</th>
                      <th className="py-2.5 px-3 font-semibold">Cache</th>
                      <th className="py-2.5 px-3 font-semibold">HITL</th>
                      <th className="py-2.5 px-3 font-semibold">Tổng độ trễ</th>
                      <th className="py-2.5 px-3 font-semibold">
                        Chi tiết Node (ms)
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {traces.length === 0 ? (
                      <tr>
                        <td
                          colSpan={8}
                          className="py-8 text-center text-slate-500"
                        >
                          {telemetryLoading
                            ? "Đang tải danh sách trace..."
                            : "Chưa có bản ghi trace nào."}
                        </td>
                      </tr>
                    ) : (
                      traces.map((tr) => (
                        <tr
                          key={tr.trace_id}
                          className="hover:bg-slate-800/40 transition align-top"
                        >
                          <td className="py-2.5 px-3 whitespace-nowrap text-slate-400">
                            {new Date(tr.created_at).toLocaleTimeString(
                              "vi-VN"
                            )}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-400">
                            {tr.thread_id}
                          </td>
                          <td className="py-2.5 px-3 max-w-xs">
                            <div className="text-slate-200 line-clamp-2">
                              {tr.question_preview}
                            </div>
                            {tr.pii_tags && tr.pii_tags.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {tr.pii_tags.map((tag, i) => (
                                  <span
                                    key={i}
                                    className="rounded bg-amber-500/15 border border-amber-500/30 px-1.5 py-0.2 text-[10px] text-amber-300"
                                  >
                                    🔒 {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </td>
                          <td className="py-2.5 px-3">
                            <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-sky-300">
                              {tr.datasource ?? "-"}
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            {tr.cache_hit ? (
                              <span className="text-emerald-400 font-semibold">
                                ⚡ HIT
                              </span>
                            ) : (
                              <span className="text-slate-500">MISS</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                                tr.hitl_status === "approved"
                                  ? "bg-emerald-500/15 text-emerald-300"
                                  : tr.hitl_status === "rejected" ||
                                      tr.hitl_status === "blocked"
                                    ? "bg-rose-500/15 text-rose-300"
                                    : tr.hitl_status === "pending"
                                      ? "bg-amber-500/15 text-amber-300"
                                      : "text-slate-500"
                              }`}
                            >
                              {tr.hitl_status}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-200 whitespace-nowrap">
                            {Math.round(tr.total_latency_ms)} ms
                          </td>
                          <td className="py-2.5 px-3">
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(tr.node_latencies || {}).map(
                                ([node, ms]) => (
                                  <span
                                    key={node}
                                    className="rounded bg-slate-950 border border-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300"
                                  >
                                    {node}: {Math.round(ms)}ms
                                  </span>
                                )
                              )}
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
