"""FastAPI Backend với endpoint POST /api/chat/stream hỗ trợ SSE Streaming (.astream_events v2)."""

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from src.agent.graph import agent_app as fallback_agent_app, compile_graph
from src.agent.state import GraphState
from src.config import CHECKPOINT_DB_PATH
from src.rag.indexing import build_or_load_vectorstore
from src.utils.pii_proxy import pii_proxy


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
) -> AsyncGenerator[str, None]:
    """Lắng nghe astream_events v2 từ LangGraph, bảo vệ bằng PIIProxy và phát SSE."""
    start_time = time.perf_counter()
    active_agent = getattr(app.state, "agent_app", None) or fallback_agent_app
    config = {"configurable": {"thread_id": thread_id}}

    # ==============================================================
    # BƯỚC 0: PII PROXY & PROMPT INJECTION GUARD (PRE-FLIGHT)
    # ==============================================================
    scan = pii_proxy.mask(question)
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
    buffered_tokens: list[str] = []

    try:
        async for event in active_agent.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event")
            name = event.get("name", "")
            metadata = event.get("metadata", {})
            langgraph_node = metadata.get("langgraph_node", "")

            if kind == "on_chain_start" and name in NODE_STATUS_MESSAGES:
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
                    if scan.has_pii:
                        buffered_tokens.append(token)
                    else:
                        yield _format_sse({"type": "token", "content": token})

            elif kind == "on_chain_end" and name == "retrieve":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and "citations" in output:
                    last_citations = output["citations"]

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
                return

        yield _format_sse(
            {
                "type": "metadata",
                "citations": last_citations,
                "hallucination_grade": last_hallucination,
                "pii_redacted": scan.detected_types,
                "thread_id": thread_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
            }
        )
        yield _format_sse({"type": "done"})

    except Exception as exc:
        yield _format_sse({"type": "error", "message": str(exc)})


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
    return {"status": "ok"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Endpoint SSE Streaming trả về tiến trình Node và từng token câu trả lời."""
    return StreamingResponse(
        _stream_agent_events(request.question, thread_id=request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/erp/approve")
async def approve_erp_order(req: ApprovalRequest):
    """Phê duyệt (có thể kèm sửa số lượng/chiết khấu) hoặc từ chối đơn hàng đang tạm dừng."""
    agent = getattr(app.state, "agent_app", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Checkpointer chưa sẵn sàng.")

    config = {"configurable": {"thread_id": req.thread_id}}
    snapshot = await agent.aget_state(config)
    if not snapshot.next or "commit_erp_order" not in snapshot.next:
        raise HTTPException(
            status_code=400,
            detail=f"Phiên `{req.thread_id}` không có đơn hàng ERP nào đang chờ phê duyệt.",
        )

    if req.action == "approve":
        if req.modified_order:
            merged_order = {**(snapshot.values.get("pending_order") or {}), **req.modified_order}
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

    async def resume_stream():
        async for event in agent.astream(None, config=config, stream_mode="updates"):
            for node_name, delta in event.items():
                yield _format_sse(
                    {
                        "type": "node_status",
                        "node": node_name,
                        "message": f"Đang tiếp tục bước: {node_name}",
                    }
                )
                if isinstance(delta, dict) and delta.get("generation"):
                    yield _format_sse({"type": "token", "content": delta["generation"]})

        final_snapshot = await agent.aget_state(config)
        yield _format_sse(
            {
                "type": "metadata",
                "approval_status": final_snapshot.values.get("approval_status"),
                "thread_id": req.thread_id,
            }
        )
        yield _format_sse({"type": "done"})

    return StreamingResponse(resume_stream(), media_type="text/event-stream")
