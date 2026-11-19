"""Script kiểm thử End-to-End (LangGraph + Hybrid RAG + MCP Tools) với Ollama thật."""

import asyncio
from mcp_server.mcp_client_bridge import execute_mcp_tool_query
from src.agent.graph import agent_app
from src.agent.state import GraphState
from src.rag.indexing import build_or_load_vectorstore


async def run_test_case_1_mcp() -> None:
    """Test Case 1: Luồng MCP Tools (tra cứu SQLite + tính tổng tiền đơn hàng)."""
    print("=" * 70)
    print("🧪 TEST CASE 1: LUỒNG MCP TOOLS (SQLite + Calculator)")
    print("=" * 70)

    question = (
        "Kiểm tra tồn kho của mã AI-WS-X1 và tính tổng tiền nếu mua 2 chiếc với chiết khấu 10%"
    )
    print(f"❓ Câu hỏi: {question}\n")

    print("--- 1A. Kiểm thử trực tiếp qua mcp_client_bridge ---")
    mcp_direct = await execute_mcp_tool_query(question)
    print(f"🛠️ Kết quả gọi MCP Tool:\n{mcp_direct}\n")

    print("--- 1B. Kiểm thử qua LangGraph StateGraph đầy đủ ---")
    state: GraphState = {
        "question": question,
        "rewritten_query": question,
        "datasource": "vectorstore",
        "metadata_filter": None,
        "documents": [],
        "citations": [],
        "mcp_result": "",
        "generation": "",
        "hallucination_grade": "yes",
        "retry_count": 0,
    }
    async for event in agent_app.astream_events(state, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        if kind == "on_chain_end" and name == "route_question":
            out = event.get("data", {}).get("output") or {}
            print(f"🚦 [route_question] -> datasource = {out.get('datasource')}")
        elif kind == "on_chain_end" and name == "call_mcp_tools":
            out = event.get("data", {}).get("output") or {}
            print(f"🛠️ [call_mcp_tools] ->\n{out.get('mcp_result')}")
        elif kind == "on_chain_end" and name == "generate":
            out = event.get("data", {}).get("output") or {}
            print(f"\n✍️ [generate - Câu trả lời cuối cùng]:\n{out.get('generation')}\n")


async def run_test_case_2_hybrid_crag() -> None:
    """Test Case 2: Luồng Hybrid Search (BM25 + Chroma Cosine + FlashRank) -> CRAG -> Generate."""
    print("=" * 70)
    print("🧪 TEST CASE 2: LUỒNG HYBRID SEARCH + CRAG + SELF-RAG")
    print("=" * 70)

    question = "Hệ thống sử dụng FlashRank Reranker model nào và trọng số của EnsembleRetriever là bao nhiêu?"
    print(f"❓ Câu hỏi: {question}\n")

    state: GraphState = {
        "question": question,
        "rewritten_query": question,
        "datasource": "vectorstore",
        "metadata_filter": None,
        "documents": [],
        "citations": [],
        "mcp_result": "",
        "generation": "",
        "hallucination_grade": "yes",
        "retry_count": 0,
    }

    async for event in agent_app.astream_events(state, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        if kind == "on_chain_end" and name == "route_question":
            out = event.get("data", {}).get("output") or {}
            print(
                f"🚦 [route_question] -> datasource = {out.get('datasource')}, "
                f"filter = {out.get('metadata_filter')}"
            )
        elif kind == "on_chain_end" and name == "retrieve":
            out = event.get("data", {}).get("output") or {}
            docs = out.get("documents", [])
            print(f"🔍 [retrieve] -> Lấy được {len(docs)} tài liệu sau FlashRank Rerank")
        elif kind == "on_chain_end" and name == "grade_documents":
            out = event.get("data", {}).get("output") or {}
            docs = out.get("documents", [])
            print(f"⚖️ [grade_documents (CRAG)] -> Giữ lại {len(docs)} tài liệu đạt chuẩn")
        elif kind == "on_chain_end" and name == "rewrite_query":
            out = event.get("data", {}).get("output") or {}
            print(f"🔄 [rewrite_query] -> Truy vấn mới: {out.get('rewritten_query')}")
        elif kind == "on_chain_end" and name == "generate":
            out = event.get("data", {}).get("output") or {}
            print(f"\n✍️ [generate - Câu trả lời cuối cùng]:\n{out.get('generation')}\n")
        elif kind == "on_chain_end" and name == "check_hallucination":
            out = event.get("data", {}).get("output") or {}
            print(f"🛡️ [check_hallucination (Self-RAG)] -> grade = {out.get('hallucination_grade')}\n")


async def main() -> None:
    print("📦 Đang khởi tạo / nạp ChromaDB Vector Store...")
    build_or_load_vectorstore()
    await run_test_case_1_mcp()
    await run_test_case_2_hybrid_crag()


if __name__ == "__main__":
    asyncio.run(main())
