# API Contract: Local Agentic RAG & MCP Assistant (v2.0)

- **Base URL (Local Dev):** `http://localhost:8000`
- **CORS:** Đã mở sẵn (`allow_origins=["*"]`) cho `http://localhost:3000` (Next.js App Router).
- **Lệnh chạy Backend nhanh:**
  ```powershell
  cd "d:\Learn_RAG_From_Scratch\local-agentic-rag"
  .\.venv\Scripts\uvicorn.exe src.api.main:app --host 0.0.0.0 --port 8000
  ```

---

## 1. Tổng quan Endpoints

| Method | Endpoint | Mục đích | Content-Type phản hồi |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Giao diện Web Chat SSE Streaming tích hợp sẵn | `text/html` |
| `GET` | `/health` | Kiểm tra trạng thái Backend FastAPI & Cache Stats | `application/json` |
| `DELETE` | `/api/cache` | Xóa toàn bộ Semantic Cache ChromaDB | `application/json` |
| `GET` | `/api/metrics` | Thống kê hiệu năng Telemetry (requests, cache hit rate, p50/p95 latency, HITL counts) | `application/json` |
| `GET` | `/api/traces` | Danh sách traces thực thi gần nhất (mới nhất trước) | `application/json` |
| `GET` | `/api/documents` | Danh sách tài liệu đã lập chỉ mục (Manifest) | `application/json` |
| `POST` | `/api/documents/upload` | Tải lên tài liệu (.md, .txt, .pdf) & lập chỉ mục vào Vector Store | `application/json` |
| `POST` | `/api/chat/stream` | Gửi câu hỏi & nhận luồng SSE (`node_status`, `token`, `approval_required`, `metadata`, `done`) | `text/event-stream` |
| `POST` | `/api/erp/approve` | Phê duyệt (có thể kèm sửa State) hoặc từ chối đơn hàng ERP đang tạm dừng tại checkpoint SQLite | `text/event-stream` |

---

## 2. Chi tiết `GET /health` & `DELETE /api/cache`

### 2.1. `GET /health`
Dùng để hiển thị trạng thái kết nối (Online/Offline) và thống kê Semantic Cache trên thanh điều hướng Next.js.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "cache_stats": {
    "entries": 12,
    "hits": 34,
    "misses": 56,
    "bypasses": 10,
    "evictions": 0,
    "hit_rate": 0.378,
    "ttl_seconds": 3600,
    "similarity_threshold": 0.92
  }
}
```

### 2.2. `DELETE /api/cache`
Xóa sạch toàn bộ Semantic Cache ChromaDB.

**Response `200 OK`:**
```json
{
  "status": "cleared",
  "cleared": 12
}
```

---

## 2.3. Chi tiết Telemetry Endpoints (`/api/metrics` & `/api/traces`)

### `GET /api/metrics`
Thống kê hiệu năng Telemetry offline (SQLite):
```json
{
  "total_requests": 42,
  "cache_hit_rate": 0.238,
  "p50_latency_ms": 1450.5,
  "p95_latency_ms": 3200.0,
  "hitl_counts": {
    "approved": 3,
    "rejected": 1,
    "none": 38
  }
}
```

### `GET /api/traces?limit=20`
Danh sách trace gần nhất (sắp xếp giảm dần theo thời gian tạo):
```json
{
  "count": 1,
  "traces": [
    {
      "trace_id": "tr-abc12345",
      "thread_id": "session_user_001",
      "question_preview": "Kiểm tra tồn kho mã AI-WS-X1...",
      "datasource": "mcp_tools",
      "cache_hit": false,
      "pii_tags": [],
      "hitl_status": "approved",
      "total_latency_ms": 1250.4,
      "node_latencies": {
        "route_question": 15.2,
        "call_mcp_tools": 850.1
      },
      "created_at": "2026-11-19T15:30:00Z"
    }
  ]
}
```

---

## 2.4. Chi tiết Document Endpoints (`/api/documents` & `/api/documents/upload`)

### `GET /api/documents`
Lấy danh sách các tài liệu đã được lập chỉ mục kèm mã băm SHA-256 và domain:
```json
{
  "count": 2,
  "documents": [
    {
      "filename": "product_specs.md",
      "doc_domain": "product_catalog",
      "sha256": "4a5b6c7d...",
      "chunk_count": 8,
      "updated_at": "2026-11-19T14:20:00Z"
    },
    {
      "filename": "rag_architecture.md",
      "doc_domain": "system_architecture",
      "sha256": "1e2f3a4b...",
      "chunk_count": 12,
      "updated_at": "2026-11-19T14:22:00Z"
    }
  ]
}
```

### `POST /api/documents/upload`
Tải lên tài liệu mới (`.md`, `.txt`, `.pdf`) qua `multipart/form-data`:
- `file`: File nhị phân (bắt buộc, đuôi `.md`, `.txt`, `.pdf`)
- `doc_domain`: Form field chuỗi tuỳ chọn (ví dụ: `product_catalog` hoặc `system_architecture`)

**Response `200 OK`:**
```json
{
  "status": "ok",
  "filename": "new_specs.md",
  "doc_domain": "product_catalog",
  "sha256": "3c8e9f...",
  "chunk_count": 4,
  "updated_at": "2026-11-19T15:40:00Z",
  "ingest_status": "indexed",
  "document": {
    "filename": "new_specs.md",
    "doc_domain": "product_catalog",
    "sha256": "3c8e9f...",
    "chunk_count": 4,
    "updated_at": "2026-11-19T15:40:00Z",
    "status": "indexed"
  }
}
```
*Lỗi `400 Bad Request`*: Trả về khi file rỗng hoặc phần mở rộng không được hỗ trợ (chỉ chấp nhận `.md`, `.txt`, `.pdf`).


---

## 3. Chi tiết `POST /api/chat/stream` (SSE Streaming)

### 3.1. Request Body (`application/json`)
```json
{
  "question": "Kiểm tra tồn kho của mã AI-WS-X1 và tính tổng tiền nếu mua 2 chiếc với chiết khấu 10%",
  "thread_id": "session_user_001",
  "bypass_cache": false
}
```

| Trường | Kiểu | Bắt buộc | Mặc định | Mô tả |
| :--- | :--- | :---: | :---: | :--- |
| `question` | `string` | Có (`min_length=1`) | - | Câu hỏi của người dùng |
| `thread_id` | `string` | Không | `"default_session"` | Mã phiên hội thoại dùng cho Durable Checkpointing |
| `bypass_cache` | `boolean` | Không | `false` | Ép buộc bỏ qua Semantic Cache ChromaDB cho truy vấn này |


### 3.2. Định dạng luồng sự kiện SSE (`text/event-stream`)
Mỗi sự kiện trả về tuân theo chuẩn Server-Sent Events:
```text
data: <chuỗi_JSON>

```
Trong đó trường `"type"` phân biệt loại sự kiện:

#### A. `type: "node_status"` (Trạng thái tư duy / Bước đang chạy)
Dùng để hiển thị thanh tiến trình suy nghĩ (*Thinking Steps / Accordion*):
```json
{
  "type": "node_status",
  "node": "pii_proxy | route_question | retrieve | grade_documents | rewrite_query | call_mcp_tools | generate | check_hallucination",
  "message": "🔒 Đã ẩn danh PII (PHONE, EMAIL) trước khi xử lý."
}
```

#### B. `type: "token"` (Mảnh nội dung câu trả lời)
Phát liên tục; Frontend nối dần (`+=`) chuỗi `content` vào bong bóng tin nhắn của Assistant và render Markdown:
```json
{
  "type": "token",
  "content": "Tồn kho của mã AI-WS-X1 là **14 chiếc**..."
}
```

#### C. `type: "metadata"` (Gói thông tin kết thúc phiên stream)
Phát 1 lần ở cuối trước `done` để hiển thị badge PII/Blocked, độ trễ và danh sách nguồn trích dẫn:
```json
{
  "type": "metadata",
  "citations": [
    {
      "source": "rag_architecture.md",
      "relevance_score": 0.9985,
      "snippet": "## 2. Cơ chế Truy xuất Lai..."
    }
  ],
  "hallucination_grade": "yes",
  "pii_redacted": ["PHONE", "EMAIL"],
  "latency_ms": 14250.6
}
```

*(Nếu bị chặn bởi PII/Prompt Injection Guard, `metadata` có `"blocked": true`, `"citations": []`).*

#### D. `type: "done"` & `type: "error"`
- `{"type": "done"}`: Đánh dấu kết thúc stream.
- `{"type": "error", "message": "..."}`: Thông báo lỗi nếu có sự cố ngoại lệ.

---

## 4. Quy trình Human-in-the-Loop (Phê duyệt chốt đơn ERP)

Khi người dùng gửi yêu cầu mang tính ghi dữ liệu (ví dụ: *"Chốt đơn đặt mua 2 chiếc AI-WS-X1 chiết khấu 10%"*), LangGraph sẽ tự động tạm dừng ngay trước node `commit_erp_order` và lưu checkpoint bền vững xuống `data/langgraph_checkpoints.sqlite`.

### 4.1. Sự kiện SSE `approval_required` (phát từ `POST /api/chat/stream`)

Thay vì chạy tiếp sang `generate`, luồng stream sẽ phát sự kiện `approval_required` rồi kết thúc bằng `done`:

```text
data: {"type": "approval_required", "thread_id": "session_user_001", "next_node": "commit_erp_order", "pending_order": {"sku": "AI-WS-X1", "quantity": 2, "discount_percent": 10.0, "total_vnd": 106020000.0}, "message": "⏸️ Đơn hàng ERP đang chờ bạn xác nhận trước khi trừ kho SQLite."}

data: {"type": "done"}
```

| Trường | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `type` | `"approval_required"` | Tín hiệu để UI khóa ô chat phiên hiện tại và hiển thị Thẻ Phê duyệt Đơn hàng (Approval Card) |
| `thread_id` | `string` | Mã phiên checkpoint lưu trong SQLite (bắt buộc gửi lại khi gọi `/api/erp/approve`) |
| `next_node` | `string` | Tên node đang bị chặn (mặc định `"commit_erp_order"`) |
| `pending_order` | `object` | Dữ liệu nháp đơn hàng (`sku`, `quantity`, `discount_percent`, `total_vnd`) để hiển thị và cho phép chỉnh sửa |
| `message` | `string` | Thông điệp hướng dẫn hiển thị trên UI |

---

### 4.2. Chi tiết Endpoint `POST /api/erp/approve`

Dùng để nối tiếp đồ thị từ checkpoint đã lưu (hoạt động bình thường ngay cả khi Backend vừa khởi động lại).

#### Request Body (`application/json`)

* **Trường hợp 1 — Phê duyệt nguyên trạng hoặc kèm chỉnh sửa State:**
  ```json
  {
    "thread_id": "session_user_001",
    "action": "approve",
    "modified_order": {
      "quantity": 1,
      "discount_percent": 10.0
    }
  }
  ```
* **Trường hợp 2 — Từ chối giao dịch (nhảy qua `commit_erp_order`, không trừ kho SQLite):**
  ```json
  {
    "thread_id": "session_user_001",
    "action": "reject"
  }
  ```

| Trường | Kiểu | Bắt buộc | Mô tả |
| :--- | :--- | :---: | :--- |
| `thread_id` | `string` | Có | Khớp chính xác với `thread_id` nhận từ sự kiện `approval_required` |
| `action` | `"approve" \| "reject"` | Có | `"approve"` để thực thi trừ kho SQLite; `"reject"` để hủy và giữ nguyên kho |
| `modified_order` | `object \| null` | Không | Chỉ dùng khi `action = "approve"`; các trường truyền lên (`quantity`, `discount_percent`, `sku`) sẽ ghi đè vào `pending_order` qua `aupdate_state()` trước khi chạy `commit_erp_order` |

#### Response `200 OK` (`text/event-stream`)

Trả về cùng chuẩn SSE một dòng `data: {"type": "..."}` giống `/api/chat/stream`:

```text
data: {"type": "node_status", "node": "commit_erp_order", "message": "Đang tiếp tục bước: commit_erp_order"}

data: {"type": "node_status", "node": "generate", "message": "Đang tiếp tục bước: generate"}

data: {"type": "token", "content": "Đã chốt đơn hàng #1: 1 x AI-WS-X1 (chiết khấu 10%), tổng thanh toán 53,010,000 VNĐ."}

data: {"type": "metadata", "approval_status": "approved", "thread_id": "session_user_001"}

data: {"type": "done"}
```

#### Mã lỗi HTTP

| Status | Nguyên nhân | Cách xử lý trên Next.js |
| :--- | :--- | :--- |
| `400 Bad Request` | `thread_id` không tồn tại hoặc đơn hàng đã được duyệt/hủy trước đó | Ẩn thẻ Approval Card và hiện toast thông báo phiên đã hoàn tất |
| `422 Unprocessable Entity` | `action` không nằm trong `["approve", "reject"]` hoặc sai kiểu dữ liệu | Kiểm tra validation form trước khi gửi |

---

## 5. Mã mẫu TypeScript hoàn chỉnh cho Next.js App Router (`lib/agentClient.ts`)

Lưu ý: Không dùng `new EventSource()` vì API nhận `POST` body. Dùng `fetch` kết hợp `ReadableStream` với hàm `consumeSseStream` dùng chung cho cả `streamAgentChat` và `submitErpApproval`:

```typescript
// lib/agentClient.ts

export interface Citation {
  source: string;
  section?: string;
  score?: number;
}

export interface PendingOrder {
  sku: string;
  quantity: number;
  discount_percent: number;
  total_vnd?: number;
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

const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENT_API_URL ?? "http://localhost:8000";

/**
 * Bộ phân tách luồng SSE chuẩn `data: {"type": "..."}` dùng chung cho
 * cả `/api/chat/stream` và `/api/erp/approve`.
 */
async function consumeSseStream(
  res: Response,
  handlers: StreamHandlers
): Promise<void> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson?.detail) detail = String(errJson.detail);
    } catch {
      // Giữ nguyên mã lỗi HTTP nếu body không phải JSON
    }
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
 * Gửi câu hỏi tới LangGraph Agent qua SSE Streaming.
 */
export async function streamAgentChat(
  question: string,
  threadId: string,
  handlers: StreamHandlers,
  options?: { signal?: AbortSignal }
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        thread_id: threadId,
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
 * Phê duyệt (có thể kèm sửa số lượng / chiết khấu) hoặc Từ chối đơn hàng ERP.
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
```

---

## 6. Mẫu Component Phê duyệt nhanh trên Next.js (`ErpApprovalCard.tsx`)

```tsx
"use client";

import { useState } from "react";
import {
  ApprovalRequiredPayload,
  StreamHandlers,
  submitErpApproval,
} from "@/lib/agentClient";

interface Props {
  approval: ApprovalRequiredPayload;
  handlers: StreamHandlers;
  onResolved: () => void;
}

export function ErpApprovalCard({ approval, handlers, onResolved }: Props) {
  const [quantity, setQuantity] = useState(approval.pending_order.quantity);
  const [discount, setDiscount] = useState(approval.pending_order.discount_percent);
  const [submitting, setSubmitting] = useState(false);

  const handleDecision = async (action: "approve" | "reject") => {
    setSubmitting(true);
    await submitErpApproval(
      {
        thread_id: approval.thread_id,
        action,
        modified_order:
          action === "approve"
            ? { quantity, discount_percent: discount }
            : undefined,
      },
      {
        ...handlers,
        onDone: () => {
          handlers.onDone?.();
          onResolved();
        },
        onError: (err) => {
          handlers.onError?.(err);
          setSubmitting(false);
        },
      }
    );
  };

  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 space-y-3">
      <p className="font-semibold text-amber-900">{approval.message}</p>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <label className="block text-xs text-slate-500">Mã SKU</label>
          <input
            disabled
            value={approval.pending_order.sku}
            className="w-full rounded border bg-slate-100 px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500">Số lượng</label>
          <input
            type="number"
            min={1}
            value={quantity}
            onChange={(e) => setQuantity(Math.max(1, Number(e.target.value)))}
            className="w-full rounded border bg-white px-2 py-1"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500">Chiết khấu (%)</label>
          <input
            type="number"
            min={0}
            max={100}
            value={discount}
            onChange={(e) => setDiscount(Number(e.target.value))}
            className="w-full rounded border bg-white px-2 py-1"
          />
        </div>
      </div>
      <div className="flex gap-2">
        <button
          disabled={submitting}
          onClick={() => handleDecision("approve")}
          className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
        >
          Phê duyệt & Trừ kho
        </button>
        <button
          disabled={submitting}
          onClick={() => handleDecision("reject")}
          className="rounded-lg bg-rose-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-rose-700 disabled:opacity-50"
        >
          Từ chối
        </button>
      </div>
    </div>
  );
}
```

---

## 7. Triển khai với Docker & Docker Compose

Hệ thống hỗ trợ chạy container hóa đầy đủ qua `docker compose`:

```powershell
# Chạy Backend (cổng 8000) và Frontend (cổng 3000)
docker compose up -d --build

# Chạy kèm service Ollama Local (dùng profile `ollama`)
docker compose --profile ollama up -d --build
```

- **Backend:** `http://localhost:8000` (`/health`, `/api/chat/stream`, `/api/documents`, `/api/metrics`, `/api/traces`)
- **Frontend Next.js:** `http://localhost:3000`
- **Ollama Local (khi bật profile):** `http://localhost:11434`

