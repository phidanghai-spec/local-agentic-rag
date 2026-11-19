"""Cài đặt các Node xử lý cho Adaptive CRAG & Self-RAG trên LangGraph."""

import re
import sqlite3
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.agent.state import (
    GradeDocuments,
    GradeHallucinations,
    GraphState,
    RouteQuery,
)
from src.config import BASE_DIR, settings
from src.utils.context_assembler import DynamicContextAssembler

ORDER_COMMIT_KEYWORDS = ("chốt đơn", "đặt mua", "tạo đơn hàng", "xác nhận mua", "trừ kho")

context_assembler = DynamicContextAssembler()

GRADE_SYSTEM_PROMPT = """Bạn là bộ lọc tài liệu cho hệ thống RAG. Chỉ đánh giá dựa trên ĐOẠN TÀI LIỆU THỰC TẾ được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
1. Tách câu hỏi thành từng vế/ý nhỏ (câu có 'và', 'hoặc', dấu phẩy thường gồm nhiều vế riêng).
2. CHẤM 'yes' nếu ĐOẠN TÀI LIỆU THỰC TẾ chứa thông tin hoặc từ khóa trả lời được ÍT NHẤT MỘT VẾ của câu hỏi.
3. CHẤM 'no' nếu ĐOẠN TÀI LIỆU THỰC TẾ nói về đối tượng/sản phẩm khác hoặc không chứa thông tin cho bất kỳ vế nào. Đặc biệt, nếu câu hỏi chỉ nêu đích danh một sản phẩm cụ thể (mẫu A) và KHÔNG phải câu hỏi so sánh nhiều sản phẩm, PHẢI chấm 'no' cho các đoạn chỉ mô tả giá bán hoặc thông số của sản phẩm khác (mẫu B).
4. TRUNG THỰC TUỆT ĐỐI: Trường `matched_aspect` CHỈ ĐƯỢC phép nhắc tên khái niệm/con số có thật trong ĐOẠN TÀI LIỆU THỰC TẾ. Tuyệt đối KHÔNG sao chép nội dung từ ví dụ minh họa. Nếu không khớp, ghi đúng chuỗi 'Không có'.

VÍ DỤ MINH HỌA (CHỈ ĐỂ HIỂU CÁCH LÀM, KHÔNG DÙNG NỘI DUNG NÀY ĐỂ TRẢ LỜI):
- Câu hỏi: "Cửa hàng giao hàng trong bao lâu và phí vận chuyển nội thành là bao nhiêu?"
  Tài liệu: "[Mục: Biểu phí] Phí vận chuyển nội thành cố định 25.000đ cho mọi đơn dưới 5kg."
  Kết quả: matched_aspect="Phí vận chuyển nội thành 25.000đ" | binary_score="yes"

- Câu hỏi: "Thời gian bảo hành của máy lọc không khí mẫu K200 là mấy năm?"
  Tài liệu: "[Mục: Hướng dẫn đổi trả] Khách hàng được đổi mới trong 7 ngày nếu hộp còn nguyên tem."
  Kết quả: matched_aspect="Không có" | binary_score="no"
"""


async def retrieve_with_fallback(query: str, metadata_filter: Dict[str, Any] | None = None):
    """Lazy wrapper để tránh import Chroma/FlashRank lúc khởi động CI."""
    from src.rag.hybrid_retriever import retrieve_with_fallback as _real_retrieve

    return await _real_retrieve(query, metadata_filter=metadata_filter)


async def execute_mcp_tool_query(question: str) -> str:
    """Lazy wrapper để tránh import langchain-mcp-adapters lúc khởi động CI."""
    from mcp_server.mcp_client_bridge import execute_mcp_tool_query as _real_mcp

    return await _real_mcp(question)


def get_llm(streaming: bool = False, temperature: float | None = None) -> ChatOllama:
    """Khởi tạo ChatOllama Local (Lazy Init)."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        streaming=streaming,
    )


_get_llm = get_llm


async def route_question(state: GraphState) -> Dict[str, Any]:
    """Node 1: Phân loại câu hỏi (Logical Routing) và trích xuất Metadata Filter."""
    question = state["question"]
    llm = get_llm()

    route_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là bộ định tuyến thông minh. Phân loại câu hỏi vào đúng 1 trong 3 nhánh:\n"
                "- 'vectorstore': Hỏi về kiến trúc RAG, FlashRank, BM25, LangGraph, cấu hình phần cứng, chính sách bảo hành.\n"
                "- 'mcp_tools': Cần tra cứu số lượng tồn kho SQLite theo mã SKU, tính tiền đơn hàng/chiết khấu, hoặc đọc trực tiếp file.\n"
                "- 'direct_answer': Chào hỏi xã giao hoặc câu hỏi chung không cần dữ liệu nội bộ.",
            ),
            ("human", "{question}"),
        ]
    )
    route_result: RouteQuery = await (route_prompt | llm.with_structured_output(RouteQuery)).ainvoke(
        {"question": question}
    )

    metadata_filter: Dict[str, Any] | None = None
    if route_result.datasource == "vectorstore":
        q_lower = question.lower()
        if "rag_architecture.md" in q_lower:
            metadata_filter = {"source": "rag_architecture.md"}
        elif "product_specs.md" in q_lower or any(
            sku in question.upper() for sku in ("AI-WS-X1", "SSD-ENT-4T", "GPU-RTX-4090")
        ):
            metadata_filter = {"source": "product_specs.md"}

    return {
        "datasource": route_result.datasource,
        "metadata_filter": metadata_filter,
        "rewritten_query": question,
        "retry_count": state.get("retry_count", 0),
    }


async def retrieve(state: GraphState) -> Dict[str, Any]:
    """Node 2: Truy xuất Hybrid Search + FlashRank Filtering + Dynamic Context Assembler."""
    query = state.get("rewritten_query") or state["question"]
    metadata_filter = state.get("metadata_filter")

    raw_docs = await retrieve_with_fallback(query, metadata_filter=metadata_filter)
    formatted_context, kept_docs, metrics = context_assembler.assemble_rag_context(
        query=query,
        docs=raw_docs,
        has_tools=False,
    )
    print(
        f"[Context Assembler] Giữ lại {metrics['kept_docs_count']}/{metrics['total_input_docs']} chunks "
        f"({metrics['rag_tokens_used']}/{metrics['rag_tokens_budget']} tokens)"
    )

    citations = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "relevance_score": round(float(doc.metadata.get("relevance_score", 0.0)), 4),
            "snippet": doc.page_content[:240],
        }
        for doc in kept_docs
    ]

    return {
        "documents": kept_docs,
        "formatted_context": formatted_context,
        "context_telemetry": metrics,
        "citations": citations,
    }


async def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Node 3 (CRAG): Chấm điểm mức độ liên quan từng phần cho câu hỏi đơn và câu hỏi ghép."""
    question = state.get("rewritten_query") or state["question"]
    docs = state.get("documents", [])
    llm = get_llm(temperature=0.0)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GRADE_SYSTEM_PROMPT),
            (
                "human",
                "[CÂU HỎI THỰC TẾ]\n{question}\n\n"
                "[ĐOẠN TÀI LIỆU THỰC TẾ]\n{document}\n\n"
                "Hãy trích vế khớp có thật trong [ĐOẠN TÀI LIỆU THỰC TẾ] vào `matched_aspect` rồi quyết định `binary_score`.",
            ),
        ]
    )
    grader = grade_prompt | llm.with_structured_output(GradeDocuments)

    filtered_docs: List[Document] = []
    filtered_citations: List[Dict[str, Any]] = []
    citations = state.get("citations", [])

    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        score = float(doc.metadata.get("relevance_score", 0.0))

        try:
            res: GradeDocuments = await grader.ainvoke(
                {"question": question, "document": doc.page_content}
            )
            decision = res.binary_score.strip().lower()
            reason = res.matched_aspect
        except Exception as exc:
            decision = "yes" if score >= 0.85 else "no"
            reason = f"Fallback JSON parse error ({exc})"

        print(
            f"[grade_documents] Chunk #{idx + 1} ({source} | flashrank={score:.3f}) "
            f"-> {decision.upper()} | Lý do: {reason}"
        )

        if decision == "yes":
            filtered_docs.append(doc)
            if idx < len(citations):
                filtered_citations.append(citations[idx])

    if not filtered_docs and docs:
        top_doc = docs[0]
        if float(top_doc.metadata.get("relevance_score", 0.0)) >= 0.90:
            print("[grade_documents] Safety fallback: giữ lại top-1 chunk có điểm FlashRank cao.")
            filtered_docs.append(top_doc)
            if citations:
                filtered_citations.append(citations[0])

    return {"documents": filtered_docs, "citations": filtered_citations}


async def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """Node 4 (Query Translation): Viết lại câu truy vấn tối ưu hơn khi tài liệu chưa khớp."""
    question = state.get("rewritten_query") or state["question"]
    retry_count = state.get("retry_count", 0) + 1
    llm = get_llm()

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là chuyên gia tối ưu hóa truy vấn tìm kiếm (Query Rewriter). "
                "Hãy viết lại câu hỏi rõ ràng, giàu từ khóa kỹ thuật hơn để tìm kiếm hiệu quả trong kho tài liệu. "
                "Chỉ trả về duy nhất câu truy vấn mới.",
            ),
            ("human", "Câu hỏi ban đầu: {question}"),
        ]
    )
    better_query = await (rewrite_prompt | llm | StrOutputParser()).ainvoke({"question": question})

    return {
        "rewritten_query": better_query.strip(),
        "metadata_filter": None,  # Mở rộng phạm vi tìm kiếm khi thử lại
        "retry_count": retry_count,
    }


def _extract_pending_order(question: str) -> Dict[str, Any]:
    """Trích xuất nhanh SKU, số lượng và % chiết khấu từ câu lệnh chốt đơn."""
    sku_match = re.search(r"\b(AI-WS-X1|SSD-ENT-4T|GPU-RTX-4090)\b", question.upper())
    qty_match = re.search(r"(\d+)\s*(?:chiếc|cái|máy|bộ|cây)", question.lower())
    disc_match = re.search(r"(?:chiết khấu|giảm)\s*(\d+(?:\.\d+)?)\s*%", question.lower())

    return {
        "sku": sku_match.group(1) if sku_match else "AI-WS-X1",
        "quantity": int(qty_match.group(1)) if qty_match else 1,
        "discount_percent": float(disc_match.group(1)) if disc_match else 0.0,
    }


async def call_mcp_tools(state: GraphState) -> Dict[str, Any]:
    """Node 5: Gọi công cụ thực tế từ FastMCP Server; chuẩn bị đơn chờ duyệt nếu là lệnh chốt đơn."""
    question = state["question"]
    mcp_output = await execute_mcp_tool_query(question)

    is_commit_request = any(k in question.lower() for k in ORDER_COMMIT_KEYWORDS)
    if is_commit_request:
        pending_order = _extract_pending_order(question)
        return {
            "mcp_result": mcp_output,
            "pending_order": pending_order,
            "requires_approval": True,
            "approval_status": "pending",
        }

    return {
        "mcp_result": mcp_output,
        "pending_order": None,
        "requires_approval": False,
        "approval_status": "none",
    }


def route_after_mcp(state: GraphState) -> str:
    """Rẽ vào cổng duyệt chốt đơn nếu là giao dịch ghi; đi thẳng tới generate nếu chỉ đọc."""
    if state.get("requires_approval") and state.get("pending_order"):
        return "commit_erp_order"
    return "generate"


async def commit_erp_order(state: GraphState) -> Dict[str, Any]:
    """
    Node thực thi ghi giao dịch chốt đơn xuống data/erp_inventory.db.
    LangGraph tự động TẠM DỪNG ngay trước node này nhờ interrupt_before=['commit_erp_order'].
    """
    order = state.get("pending_order") or {}
    sku = str(order.get("sku", "AI-WS-X1")).strip().upper()
    quantity = int(order.get("quantity", 1))
    discount = float(order.get("discount_percent", 0.0))

    db_path = BASE_DIR / "data" / "erp_inventory.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS erp_orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                discount_percent REAL NOT NULL,
                total_vnd REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        row = cur.execute(
            "SELECT stock_qty, unit_price_vnd FROM inventory WHERE sku = ?",
            (sku,),
        ).fetchone()
        if not row:
            return {
                "approval_status": "rejected",
                "requires_approval": False,
                "mcp_result": f"Không tìm thấy mã sản phẩm {sku} trong kho ERP.",
            }

        current_stock, unit_price = int(row[0]), float(row[1])
        if current_stock < quantity:
            return {
                "approval_status": "rejected",
                "requires_approval": False,
                "mcp_result": f"Tồn kho không đủ (hiện còn {current_stock}, yêu cầu {quantity}).",
            }

        final_total = round(unit_price * quantity * (1.0 - discount / 100.0), 2)
        cur.execute(
            "UPDATE inventory SET stock_qty = stock_qty - ? WHERE sku = ?",
            (quantity, sku),
        )
        cur.execute(
            "INSERT INTO erp_orders (sku, quantity, discount_percent, total_vnd, status) VALUES (?, ?, ?, ?, ?)",
            (sku, quantity, discount, final_total, "COMMITTED"),
        )
        order_id = cur.lastrowid
        conn.commit()

        summary = (
            f"Đã chốt đơn hàng #{order_id}: {quantity} x {sku} "
            f"(chiết khấu {discount}%), tổng thanh toán {int(final_total):,} VNĐ. "
            f"Tồn kho còn lại: {current_stock - quantity} chiếc."
        )
        return {
            "approval_status": "approved",
            "requires_approval": False,
            "mcp_result": summary,
        }
    finally:
        conn.close()


async def generate(state: GraphState) -> Dict[str, Any]:
    """Node 6: Tổng hợp câu trả lời cuối cùng (hỗ trợ streaming token)."""
    question = state["question"]
    datasource = state.get("datasource", "vectorstore")
    docs = state.get("documents", [])
    mcp_result = state.get("mcp_result", "")

    llm = get_llm(streaming=True)

    if datasource == "mcp_tools" and mcp_result:
        context_text = f"Kết quả từ hệ thống công cụ MCP:\n{mcp_result}"
    elif docs:
        context_text, _, _ = context_assembler.assemble_rag_context(
            query=question,
            docs=docs,
            has_tools=False,
        )
    else:
        context_text = "Không có tài liệu nội bộ phù hợp. Hãy trả lời ngắn gọn, trung thực."

    gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là Trợ lý Kỹ thuật AI chính xác và trung thực. "
                "Chỉ sử dụng thông tin trong phần Ngữ cảnh được cung cấp để trả lời bằng tiếng Việt rõ ràng. "
                "Nếu ngữ cảnh không chứa thông tin, hãy nói rõ bạn không tìm thấy dữ liệu trong tài liệu.",
            ),
            ("human", "Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"),
        ]
    )

    generation = await (gen_prompt | llm | StrOutputParser()).ainvoke(
        {"context": context_text, "question": question}
    )

    return {"generation": generation}


async def check_hallucination(state: GraphState) -> Dict[str, Any]:
    """Node 7 (Self-RAG): Kiểm tra câu trả lời có bám sát tài liệu gốc hay bịa đặt."""
    docs = state.get("documents", [])
    generation = state.get("generation", "")

    if not docs:
        return {"hallucination_grade": "yes"}

    llm = get_llm()
    facts = "\n\n".join(d.page_content for d in docs)

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là bộ kiểm định chống ảo giác (Self-RAG). Đánh giá xem câu trả lời có được "
                "hỗ trợ bởi tập tài liệu gốc hay không. Trả về 'yes' nếu bám sát sự thật, 'no' nếu bịa đặt.",
            ),
            ("human", "Tài liệu gốc:\n{documents}\n\nCâu trả lời của AI:\n{generation}"),
        ]
    )

    grade: GradeHallucinations = await (
        hallucination_prompt | llm.with_structured_output(GradeHallucinations)
    ).ainvoke({"documents": facts, "generation": generation})

    return {"hallucination_grade": grade.binary_score}
