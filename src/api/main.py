"""FastAPI Backend với endpoint POST /api/chat/stream hỗ trợ SSE Streaming (.astream_events v2)."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from src.agent.graph import agent_app as fallback_agent_app, compile_graph
from src.agent.state import GraphState
from src.config import BASE_DIR, CHECKPOINT_DB_PATH, settings
from src.rag.indexing import build_or_load_vectorstore, ingest_file, list_indexed_documents

from src.utils.pii_proxy import pii_proxy
from src.utils.semantic_cache import SemanticCache, should_bypass_cache
from src.utils.telemetry import TelemetryStore

CACHE_DB_PATH = BASE_DIR / "chroma_cache_db"
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Khởi tạo lười (lazy) Semantic Cache dùng chung cho toàn server."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache(persist_directory=CACHE_DB_PATH)
    return _semantic_cache


_telemetry_store: Optional[TelemetryStore] = None
_approvals_in_flight: set[str] = set()


def get_telemetry_store() -> TelemetryStore:
    """Khởi tạo lười TelemetryStore (SQLite offline) dùng chung cho toàn server."""
    global _telemetry_store
    if _telemetry_store is None:
        _telemetry_store = TelemetryStore()
    return _telemetry_store


async def _record_trace(**kwargs: Any) -> None:
    """Ghi trace; telemetry hỏng không bao giờ được làm sập luồng chat."""
    try:
        await asyncio.to_thread(get_telemetry_store().record_trace, **kwargs)
    except Exception as exc:
        print(f"[Telemetry] Lỗi ghi trace, bỏ qua: {exc}")


NODE_STATUS_MESSAGES: Dict[str, str] = {
    "route_question": "Đang phân tích và định tuyến câu hỏi (Logical Router)...",
    "retrieve": "Đang truy xuất Hybrid Search (BM25 + ChromaDB) & FlashRank Reranking...",
    "grade_documents": "Đang thẩm định độ chính xác tài liệu (CRAG Grader)...",
    "rewrite_query": "Tài liệu chưa khớp -> Đang tối ưu lại câu truy vấn (Query Rewriting)...",
    "call_mcp_tools": "Đang gọi công cụ từ MCP Server...",
    "commit_erp_order": "Đang ghi giao dịch chốt đơn xuống SQLite...",
    "generate": "Đang tổng hợp câu trả lời...",
    "check_hallucination": "Đang kiểm tra chống ảo giác (Self-RAG)...",
}


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Khởi tạo ChromaDB và AsyncSqliteSaver Durable Checkpointer khi bật server."""
    build_or_load_vectorstore()
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        await checkpointer.setup()
        fastapi_app.state.agent_app = compile_graph(checkpointer=checkpointer, enable_hitl=True)
        print(f"[Lifespan] Đã kích hoạt AsyncSqliteSaver tại: {CHECKPOINT_DB_PATH}")
        yield
    print("[Lifespan] Đã đóng kết nối SQLite Checkpointer an toàn.")


app = FastAPI(
    title="Local Agentic RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Schema yêu cầu chat từ Frontend Next.js."""

    question: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    thread_id: str = Field(default="default_session", description="Mã phiên hội thoại bền vững")
    bypass_cache: bool = Field(default=False, description="Bỏ qua Semantic Cache cho yêu cầu này")


class ApprovalRequest(BaseModel):
    """Schema phê duyệt hoặc từ chối chốt đơn ERP (Human-in-the-loop)."""

    thread_id: str
    action: Literal["approve", "reject"]
    modified_order: Optional[Dict[str, Any]] = None


def _format_sse(payload: dict) -> str:
    """Đóng gói JSON theo chuẩn Server-Sent Events (SSE)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_agent_events(
    question: str,
    thread_id: str = "default_session",
    bypass_cache: bool = False,
) -> AsyncGenerator[str, None]:
    """Lắng nghe astream_events v2 từ LangGraph, bảo vệ bằng PIIProxy và phát SSE."""
    start_time = time.perf_counter()
    active_agent = getattr(app.state, "agent_app", None) or fallback_agent_app
    config = {"configurable": {"thread_id": thread_id}}
    node_latencies: Dict[str, float] = {}
    node_started: Dict[str, float] = {}
    trace_question = question
    trace_pii: list[str] = []

    async def _trace(datasource: str, cache_hit: bool = False, hitl_status: str = "none") -> None:
        await _record_trace(
            thread_id=thread_id,
            question=trace_question,
            datasource=datasource,
            cache_hit=cache_hit,
            pii_tags=trace_pii,
            hitl_status=hitl_status,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            node_latencies=node_latencies,
        )

    # ==============================================================
    # BƯỚC 0: PII PROXY & PROMPT INJECTION GUARD (PRE-FLIGHT)
    # ==============================================================
    scan = pii_proxy.mask(question)
    # Chỉ lưu bản đã ẩn danh vào telemetry (không rò PII xuống đĩa).
    trace_question = scan.masked_text
    trace_pii = list(scan.detected_types)
    if scan.is_injection_blocked:
        yield _format_sse(
            {
                "type": "node_status",
                "node": "pii_proxy",
                "message": f"🛡️ Đã chặn yêu cầu không an toàn: {scan.injection_reason}",
            }
        )
        yield _format_sse(
            {
                "type": "token",
                "content": "Yêu cầu bị từ chối bởi lớp bảo mật do chứa chỉ thị không hợp lệ.",
            }
        )
        yield _format_sse(
            {
                "type": "metadata",
                "citations": [],
                "blocked": True,
                "pii_redacted": [],
            }
        )
        yield _format_sse({"type": "done"})
        await _trace("blocked", hitl_status="blocked")
        return

    if scan.has_pii:
        yield _format_sse(
            {
                "type": "node_status",
                "node": "pii_proxy",
                "message": f"🔒 Đã ẩn danh PII ({', '.join(scan.detected_types)}) trước khi xử lý.",
            }
        )

    safe_question = scan.masked_text

    # ==============================================================
    # BƯỚC 0.5: SEMANTIC CACHE LOOKUP (bỏ qua ERP/ghi/PII hoặc khi client yêu cầu)
    # ==============================================================
    cache = get_semantic_cache()
    cache_allowed = not bypass_cache and not should_bypass_cache(question)
    if cache_allowed:
        try:
            cached = await asyncio.to_thread(cache.get, safe_question)
        except Exception as exc:  # cache hỏng không được làm sập luồng chat
            print(f"[SemanticCache] Lỗi tra cứu, bỏ qua cache: {exc}")
            cached = None
        if cached:
            yield _format_sse(
                {
                    "type": "node_status",
                    "node": "semantic_cache",
                    "message": f"⚡ Cache HIT (similarity={cached['similarity']}) - trả lời ngay từ Semantic Cache.",
                }
            )
            yield _format_sse({"type": "token", "content": cached["response"]})
            yield _format_sse(
                {
                    "type": "metadata",
                    "citations": cached["citations"],
                    "cache_hit": True,
                    "similarity": cached["similarity"],
                    "pii_redacted": [],
                    "thread_id": thread_id,
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
                }
            )
            yield _format_sse({"type": "done"})
            await _trace("semantic_cache", cache_hit=True)
            return
    else:
        cache.record_bypass()

    initial_state: GraphState = {
        "question": safe_question,
        "rewritten_query": safe_question,
        "datasource": "vectorstore",
        "metadata_filter": None,
        "documents": [],
        "formatted_context": "",
        "context_telemetry": {},
        "citations": [],
        "mcp_result": "",
        "pending_order": None,
        "requires_approval": False,
        "approval_status": "none",
        "generation": "",
        "hallucination_grade": "yes",
        "retry_count": 0,
    }

    last_citations = []
    last_hallucination = "yes"
    last_datasource = "vectorstore"
    last_generation = ""
    buffered_tokens: list[str] = []
    all_tokens: list[str] = []

    try:
        async for event in active_agent.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event")
            name = event.get("name", "")
            metadata = event.get("metadata", {})
            langgraph_node = metadata.get("langgraph_node", "")

            if kind == "on_chain_end" and name in node_started:
                # Cộng dồn thời gian node (vòng rewrite_query -> retrieve có thể chạy lại nhiều lần).
                elapsed_ms = (time.perf_counter() - node_started.pop(name)) * 1000
                node_latencies[name] = node_latencies.get(name, 0.0) + elapsed_ms

            if kind == "on_chain_start" and name in NODE_STATUS_MESSAGES:
                node_started[name] = time.perf_counter()
                yield _format_sse(
                    {
                        "type": "node_status",
                        "node": name,
                        "message": NODE_STATUS_MESSAGES[name],
                    }
                )

            elif kind == "on_chat_model_stream" and langgraph_node == "generate":
                chunk = event.get("data", {}).get("chunk")
                token = getattr(chunk, "content", "") if chunk else ""
                if token:
                    all_tokens.append(token)
                    if scan.has_pii:
                        buffered_tokens.append(token)
                    else:
                        yield _format_sse({"type": "token", "content": token})

            elif kind == "on_chain_end" and name == "retrieve":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and "citations" in output:
                    last_citations = output["citations"]

            elif kind == "on_chain_end" and name == "route_question":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and output.get("datasource"):
                    last_datasource = output["datasource"]

            elif kind == "on_chain_end" and name == "generate":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and output.get("generation"):
                    last_generation = output["generation"]

            elif kind == "on_chain_end" and name == "check_hallucination":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and "hallucination_grade" in output:
                    last_hallucination = output["hallucination_grade"]

        if scan.has_pii and buffered_tokens:
            restored_answer = pii_proxy.unmask("".join(buffered_tokens), scan.vault)
            yield _format_sse({"type": "token", "content": restored_answer})

        if getattr(app.state, "agent_app", None) is not None:
            snapshot = await active_agent.aget_state(config)
            if snapshot.next and "commit_erp_order" in snapshot.next:
                pending_order = snapshot.values.get("pending_order", {})
                yield _format_sse(
                    {
                        "type": "approval_required",
                        "thread_id": thread_id,
                        "next_node": "commit_erp_order",
                        "pending_order": pending_order,
                        "message": "⏸️ Đơn hàng ERP đang chờ bạn xác nhận trước khi trừ kho SQLite.",
                    }
                )
                yield _format_sse({"type": "done"})
                await _trace(last_datasource, hitl_status="pending")
                return

        yield _format_sse(
            {
                "type": "metadata",
                "citations": last_citations,
                "hallucination_grade": last_hallucination,
                "pii_redacted": scan.detected_types,
                "cache_hit": False,
                "similarity": None,
                "thread_id": thread_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
            }
        )
        yield _format_sse({"type": "done"})
        await _trace(last_datasource)

        # Ghi cache sau khi đã stream xong (không chặn người dùng); chỉ cache câu trả lời đã grounded.
        answer = (last_generation or "".join(all_tokens)).strip()
        if (
            cache_allowed
            and answer
            and last_hallucination == "yes"
            and not should_bypass_cache(question, last_datasource)
        ):
            try:
                await asyncio.to_thread(cache.set, safe_question, answer, last_citations)
            except Exception as exc:
                print(f"[SemanticCache] Lỗi ghi cache: {exc}")

    except Exception as exc:
        yield _format_sse({"type": "error", "message": str(exc)})
        await _trace(last_datasource, hitl_status="error")


@app.get("/", response_class=HTMLResponse)
async def web_chat_ui():
    """Giao diện Web Chat SSE Streaming trực tiếp để kiểm thử Agent thời gian thực."""
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Local Agentic RAG & MCP Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 flex items-center justify-between">
    <div>
      <h1 class="text-lg font-bold text-sky-400">Local Agentic RAG &amp; MCP Assistant</h1>
      <p class="text-xs text-slate-400">LangGraph CRAG + Hybrid Search (BM25 + Chroma Cosine + FlashRank) + FastMCP + Ollama</p>
    </div>
    <span class="inline-flex items-center gap-2 text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
      <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> SSE Streaming Ready
    </span>
  </header>

  <main class="flex-1 max-w-4xl w-full mx-auto p-6 flex flex-col gap-5">
    <section class="bg-slate-900 border border-slate-800 rounded-xl p-4">
      <p class="text-xs uppercase tracking-wider text-slate-400 mb-3 font-semibold">Câu hỏi gợi ý kiểm thử nhanh</p>
      <div class="flex flex-wrap gap-2">
        <button onclick="useSample(this.dataset.q)" data-q="Kiểm tra tồn kho của mã AI-WS-X1 và tính tổng tiền nếu mua 2 chiếc với chiết khấu 10%"
          class="text-left text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-amber-300 px-3 py-2 rounded-lg transition">
          🛠️ MCP ERP: Tồn kho AI-WS-X1 + Tính tiền 2 chiếc (-10%)
        </button>
        <button onclick="useSample(this.dataset.q)" data-q="Hệ thống sử dụng mô hình Reranker nào và chạy trên phần cứng nào?"
          class="text-left text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sky-300 px-3 py-2 rounded-lg transition">
          🔍 Hybrid RAG: Mô hình Reranker &amp; phần cứng chạy
        </button>
        <button onclick="useSample(this.dataset.q)" data-q="Laptop AI Workstation X1 có giá niêm yết và thời gian bảo hành bao lâu?"
          class="text-left text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-emerald-300 px-3 py-2 rounded-lg transition">
          📄 Product Specs: Giá &amp; bảo hành Workstation X1
        </button>
      </div>
    </section>

    <section id="chatFeed" class="flex-1 space-y-4 overflow-y-auto"></section>

    <form id="chatForm" class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex gap-3 sticky bottom-4">
      <input id="questionInput" type="text" required placeholder="Nhập câu hỏi cho Local Agentic RAG..."
        class="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-sky-500" />
      <button id="sendBtn" type="submit"
        class="bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-sm font-medium px-5 py-2.5 rounded-lg transition">
        Gửi
      </button>
    </form>
  </main>

  <script>
    const chatFeed = document.getElementById('chatFeed');
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');

    const NODE_ICONS = {
      route_question: '🚦',
      retrieve: '🔍',
      grade_documents: '⚖️',
      rewrite_query: '🔄',
      call_mcp_tools: '🛠️',
      generate: '✍️',
      check_hallucination: '🛡️'
    };

    function useSample(text) {
      questionInput.value = text;
      questionInput.focus();
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = questionInput.value.trim();
      if (!question) return;

      questionInput.value = '';
      sendBtn.disabled = true;

      const userBubble = document.createElement('div');
      userBubble.className = 'flex justify-end';
      userBubble.innerHTML = `<div class="bg-sky-600 text-white text-sm px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-2xl">${escapeHtml(question)}</div>`;
      chatFeed.appendChild(userBubble);

      const botCard = document.createElement('div');
      botCard.className = 'bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3';
      botCard.innerHTML = `
        <div class="border-b border-slate-800 pb-2">
          <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Agent Thought Process (SSE Live)</div>
          <ul class="thought-list space-y-1 text-xs text-slate-300"></ul>
        </div>
        <div class="answer-box text-sm leading-relaxed whitespace-pre-wrap text-slate-100"></div>
        <div class="citations-box hidden pt-2 border-t border-slate-800 space-y-2"></div>
      `;
      chatFeed.appendChild(botCard);

      const thoughtList = botCard.querySelector('.thought-list');
      const answerBox = botCard.querySelector('.answer-box');
      const citationsBox = botCard.querySelector('.citations-box');

      try {
        const res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const parts = buffer.split('\\n\\n');
          buffer = parts.pop() || '';

          for (const part of parts) {
            const line = part.trim();
            if (!line.startsWith('data:')) continue;
            const payload = JSON.parse(line.slice(5).trim());

            if (payload.type === 'node_status') {
              const li = document.createElement('li');
              const icon = NODE_ICONS[payload.node] || '⚡';
              li.textContent = `${icon} [${payload.node}] ${payload.message}`;
              thoughtList.appendChild(li);
            } else if (payload.type === 'token') {
              answerBox.textContent += payload.content;
            } else if (payload.type === 'metadata' && payload.citations && payload.citations.length) {
              citationsBox.classList.remove('hidden');
              citationsBox.innerHTML = `<div class="text-xs font-semibold text-slate-400">Nguồn trích dẫn (Self-RAG grounded: ${payload.hallucination_grade}):</div>` +
                payload.citations.map(c => `<div class="text-xs bg-slate-950 border border-slate-800 rounded p-2"><span class="text-sky-400 font-medium">${escapeHtml(c.source)}</span> (FlashRank score: ${c.relevance_score})<div class="text-slate-400 mt-1">${escapeHtml(c.snippet)}...</div></div>`).join('');
            } else if (payload.type === 'error') {
              answerBox.innerHTML = `<span class="text-rose-400">Lỗi: ${escapeHtml(payload.message)}</span>`;
            }
          }
          window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
      } catch (err) {
        answerBox.innerHTML = `<span class="text-rose-400">Lỗi kết nối: ${escapeHtml(String(err))}</span>`;
      } finally {
        sendBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái máy chủ."""
    try:
        stats = get_semantic_cache().get_stats()
    except Exception as exc:
        stats = {"error": str(exc)}
    return {"status": "ok", "cache_stats": stats}


@app.delete("/api/cache")
async def clear_cache():
    """Xóa toàn bộ Semantic Cache."""
    try:
        cleared = await asyncio.to_thread(get_semantic_cache().clear)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể xóa cache: {exc}")
    return {"status": "cleared", "cleared": cleared}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Endpoint SSE Streaming trả về tiến trình Node và từng token câu trả lời."""
    return StreamingResponse(
        _stream_agent_events(
            request.question,
            thread_id=request.thread_id,
            bypass_cache=request.bypass_cache,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/metrics")
async def get_metrics():
    """Số liệu quan sát offline: số request, cache hit rate, p50/p95 latency, thống kê HITL."""
    try:
        return await asyncio.to_thread(get_telemetry_store().get_metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không đọc được telemetry: {exc}")


@app.get("/api/traces")
async def get_traces(limit: int = 20):
    """Danh sách trace gần nhất (mới nhất trước)."""
    try:
        traces = await asyncio.to_thread(get_telemetry_store().list_traces, limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không đọc được telemetry: {exc}")
    return {"count": len(traces), "traces": traces}


@app.get("/api/documents")
async def get_documents():
    """Lấy danh sách các tài liệu đã được lập chỉ mục trong manifest."""
    try:
        docs = await asyncio.to_thread(list_indexed_documents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không đọc được danh sách tài liệu: {exc}")
    return {"count": len(docs), "documents": docs}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_domain: Optional[str] = Form(default=None),
):
    """Tải lên tài liệu (.md, .txt, .pdf) và lập chỉ mục vào Vector Store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ.")

    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()
    if ext not in {".md", ".txt", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng tệp không được hỗ trợ: {ext}. Hệ thống chỉ hỗ trợ .md, .txt, .pdf.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Tệp tải lên rỗng.")

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        file_path = settings.data_dir / filename
        with open(file_path, "wb") as f:
            f.write(contents)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể lưu tệp: {exc}")

    try:
        result = await asyncio.to_thread(
            ingest_file,
            file_path=file_path,
            doc_domain=doc_domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi lập chỉ mục tài liệu: {exc}")

    return {
        "status": "ok",
        "filename": result.get("filename", filename),
        "doc_domain": result.get("doc_domain"),
        "sha256": result.get("sha256"),
        "chunk_count": result.get("chunk_count", 0),
        "updated_at": result.get("updated_at"),
        "ingest_status": result.get("status"),
        "document": result,
    }



@app.post("/api/erp/approve")
async def approve_erp_order(req: ApprovalRequest):
    """Phê duyệt (có thể kèm sửa số lượng/chiết khấu) hoặc từ chối đơn hàng đang tạm dừng."""
    agent = getattr(app.state, "agent_app", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Checkpointer chưa sẵn sàng.")

    if req.thread_id in _approvals_in_flight:
        raise HTTPException(
            status_code=409,
            detail={
                "status": "in_progress",
                "message": f"Phiên `{req.thread_id}` đang được xử lý phê duyệt, không thể gửi trùng.",
            },
        )

    config = {"configurable": {"thread_id": req.thread_id}}
    snapshot = await agent.aget_state(config)
    if not snapshot.next or "commit_erp_order" not in snapshot.next:
        resolved_status = (snapshot.values or {}).get("approval_status")
        if resolved_status in ("approved", "rejected"):
            # Idempotent: đơn đã được xử lý xong -> không resume/trừ kho lần nữa.
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "already_resolved",
                    "approval_status": resolved_status,
                    "message": f"Phiên `{req.thread_id}` đã được xử lý ({resolved_status}); bỏ qua yêu cầu trùng.",
                },
            )
        raise HTTPException(
            status_code=400,
            detail=f"Phiên `{req.thread_id}` không có đơn hàng ERP nào đang chờ phê duyệt.",
        )

    _approvals_in_flight.add(req.thread_id)
    start_time = time.perf_counter()
    try:
        pending_order = (snapshot.values or {}).get("pending_order") or {}
        if req.action == "approve":
            if req.modified_order:
                # Giữ nguyên idempotency_key đã cấp lúc tạo đơn, kể cả khi client sửa số lượng/chiết khấu.
                merged_order = {**pending_order, **req.modified_order}
                if pending_order.get("idempotency_key"):
                    merged_order["idempotency_key"] = pending_order["idempotency_key"]
                await agent.aupdate_state(
                    config,
                    {"pending_order": merged_order, "approval_status": "approved"},
                )
        else:
            await agent.aupdate_state(
                config,
                {
                    "approval_status": "rejected",
                    "requires_approval": False,
                    "mcp_result": "Người dùng đã TỪ CHỐI phê duyệt đơn hàng. Không có thay đổi nào trên kho ERP.",
                },
                as_node="commit_erp_order",
            )
    except Exception:
        _approvals_in_flight.discard(req.thread_id)
        raise

    async def resume_stream():
        node_latencies: Dict[str, float] = {}
        last_mark = time.perf_counter()
        try:
            async for event in agent.astream(None, config=config, stream_mode="updates"):
                now = time.perf_counter()
                for node_name, delta in event.items():
                    node_latencies[node_name] = node_latencies.get(node_name, 0.0) + (now - last_mark) * 1000
                    yield _format_sse(
                        {
                            "type": "node_status",
                            "node": node_name,
                            "message": f"Đang tiếp tục bước: {node_name}",
                        }
                    )
                    if isinstance(delta, dict) and delta.get("generation"):
                        yield _format_sse({"type": "token", "content": delta["generation"]})
                last_mark = now

            final_snapshot = await agent.aget_state(config)
            final_status = final_snapshot.values.get("approval_status")
            yield _format_sse(
                {
                    "type": "metadata",
                    "approval_status": final_status,
                    "thread_id": req.thread_id,
                }
            )
            yield _format_sse({"type": "done"})
            await _record_trace(
                thread_id=req.thread_id,
                question=f"[ERP {req.action}] {pending_order.get('sku', '')} x{pending_order.get('quantity', '')}",
                datasource="mcp_tools",
                cache_hit=False,
                pii_tags=[],
                hitl_status=final_status or ("approved" if req.action == "approve" else "rejected"),
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                node_latencies=node_latencies,
            )
        finally:
            _approvals_in_flight.discard(req.thread_id)

    return StreamingResponse(resume_stream(), media_type="text/event-stream")
