// frontend/lib/agentClient.ts

export interface Citation {
  source: string;
  relevance_score?: number;
  score?: number;
  snippet?: string;
  section?: string;
}

export interface PendingOrder {
  sku: string;
  quantity: number;
  discount_percent: number;
  total_vnd?: number;
  unit_price_vnd?: number;
  product_name?: string;
  idempotency_key?: string;
}

export interface ApprovalRequiredPayload {
  thread_id: string;
  next_node: string;
  pending_order: PendingOrder;
  message: string;
}

export interface StreamMetadata {
  citations?: Citation[];
  hallucination_grade?: "yes" | "no";
  pii_redacted?: string[];
  blocked?: boolean;
  cache_hit?: boolean;
  similarity?: number | null;
  approval_status?: "none" | "pending" | "approved" | "rejected";
  thread_id?: string;
  executed_nodes?: string[];
  latency_ms?: number;
}

export interface ApproveOrderRequest {
  thread_id: string;
  action: "approve" | "reject";
  modified_order?: Partial<PendingOrder>;
}

export interface StreamHandlers {
  onStatus?: (node: string, message: string) => void;
  onToken?: (chunk: string) => void;
  onApprovalRequired?: (payload: ApprovalRequiredPayload) => void;
  onMetadata?: (meta: StreamMetadata) => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
}

export interface CacheStats {
  hits?: number;
  misses?: number;
  bypasses?: number;
  sets?: number;
  evictions?: number;
  hit_rate?: number;
  entry_count?: number;
  threshold?: number;
  ttl_seconds?: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  cache_stats?: CacheStats;
}

export interface ClearCacheResponse {
  status: string;
  cleared: number;
}

export interface TelemetryMetrics {
  total_requests: number;
  cache_hit_rate: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  hitl_counts: Record<string, number>;
}

export interface TraceRecord {
  trace_id: string;
  thread_id: string;
  question_preview: string;
  datasource: string | null;
  cache_hit: boolean;
  pii_tags: string[];
  hitl_status: string;
  total_latency_ms: number;
  node_latencies: Record<string, number>;
  created_at: string;
}

export interface TracesResponse {
  count: number;
  traces: TraceRecord[];
}

export interface IndexedDocument {
  filename: string;
  doc_domain: string;
  sha256: string;
  chunk_count: number;
  updated_at: string;
  status?: string;
}

export interface DocumentsResponse {
  count: number;
  documents: IndexedDocument[];
}

export interface UploadDocumentResponse {
  status: string;
  filename: string;
  doc_domain?: string;
  sha256?: string;
  chunk_count: number;
  updated_at?: string;
  ingest_status?: string;
  document?: IndexedDocument;
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENT_API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

async function extractErrorDetail(res: Response): Promise<string> {
  let detail = `HTTP ${res.status}`;
  try {
    const errJson = await res.json();
    if (errJson?.detail) {
      detail =
        typeof errJson.detail === "string"
          ? errJson.detail
          : errJson.detail.message || JSON.stringify(errJson.detail);
    }
  } catch {
    // Keep HTTP status code when response body is not JSON
  }
  return detail;
}

/**
 * Shared SSE stream parser (`data: {"type": "..."}`) for
 * both `/api/chat/stream` and `/api/erp/approve`.
 */
async function consumeSseStream(
  res: Response,
  handlers: StreamHandlers
): Promise<void> {
  if (!res.ok) {
    const detail = await extractErrorDetail(res);
    throw new Error(`Backend error: ${detail}`);
  }

  if (!res.body) {
    throw new Error("Trình duyệt không nhận được ReadableStream từ Backend.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawBlock of events) {
      const dataLines = rawBlock
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());

      if (dataLines.length === 0) continue;

      const payload = JSON.parse(dataLines.join("\n"));

      switch (payload.type) {
        case "node_status":
          handlers.onStatus?.(payload.node ?? "", payload.message ?? "");
          break;
        case "token":
          handlers.onToken?.(payload.content ?? "");
          break;
        case "approval_required":
          handlers.onApprovalRequired?.({
            thread_id: payload.thread_id,
            next_node: payload.next_node,
            pending_order: payload.pending_order,
            message: payload.message,
          });
          break;
        case "metadata":
          handlers.onMetadata?.(payload as StreamMetadata);
          break;
        case "done":
          handlers.onDone?.();
          break;
        case "error":
          throw new Error(payload.message || "Stream error");
      }
    }
  }
}

/**
 * Send a question to the LangGraph Agent over SSE Streaming (`POST /api/chat/stream`).
 */
export async function streamAgentChat(
  question: string,
  threadId: string,
  handlers: StreamHandlers,
  options?: { bypassCache?: boolean; signal?: AbortSignal }
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        thread_id: threadId,
        bypass_cache: Boolean(options?.bypassCache),
      }),
      signal: options?.signal,
    });
    await consumeSseStream(res, handlers);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}

/**
 * Approve (with optional modified quantity / discount_percent) or Reject a pending ERP order (`POST /api/erp/approve`).
 */
export async function submitErpApproval(
  request: ApproveOrderRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/erp/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
    await consumeSseStream(res, handlers);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}

/**
 * Check FastAPI Backend health and Semantic Cache stats (`GET /health`).
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  return (await res.json()) as HealthResponse;
}

/**
 * Clear the Semantic Cache (`DELETE /api/cache`).
 */
export async function clearSemanticCache(): Promise<ClearCacheResponse> {
  const res = await fetch(`${API_BASE_URL}/api/cache`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  return (await res.json()) as ClearCacheResponse;
}

/**
 * Fetch offline telemetry summary metrics (`GET /api/metrics`).
 */
export async function fetchMetrics(): Promise<TelemetryMetrics> {
  const res = await fetch(`${API_BASE_URL}/api/metrics`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  return (await res.json()) as TelemetryMetrics;
}

/**
 * Fetch recent execution traces (`GET /api/traces`).
 */
export async function fetchTraces(limit = 20): Promise<TracesResponse> {
  const res = await fetch(`${API_BASE_URL}/api/traces?limit=${limit}`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  const data = await res.json();
  if (Array.isArray(data)) {
    return { count: data.length, traces: data as TraceRecord[] };
  }
  return data as TracesResponse;
}

/**
 * Fetch indexed documents from ingestion manifest (`GET /api/documents`).
 */
export async function fetchDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/documents`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  const data = await res.json();
  if (Array.isArray(data)) {
    return { count: data.length, documents: data as IndexedDocument[] };
  }
  return data as DocumentsResponse;
}

/**
 * Upload and index a document (`.md`, `.txt`, `.pdf`) with optional `doc_domain` (`POST /api/documents/upload`).
 */
export async function uploadDocument(
  file: File,
  docDomain?: string
): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (docDomain && docDomain.trim().length > 0) {
    formData.append("doc_domain", docDomain.trim());
  }

  const res = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(await extractErrorDetail(res));
  }
  return (await res.json()) as UploadDocumentResponse;
}
