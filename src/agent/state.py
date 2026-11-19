"""Định nghĩa GraphState và các Pydantic v2 Schemas cho Routing, CRAG, Self-RAG."""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Định tuyến câu hỏi người dùng đến nguồn dữ liệu hoặc công cụ tối ưu."""

    datasource: Literal["vectorstore", "mcp_tools", "direct_answer"] = Field(
        ...,
        description=(
            "Chọn 'vectorstore' nếu câu hỏi về tài liệu kỹ thuật, kiến trúc RAG, thông số hoặc chính sách sản phẩm; "
            "'mcp_tools' nếu cần tra cứu tồn kho SQLite theo SKU, tính toán hóa đơn/chiết khấu, hoặc đọc nguyên file; "
            "'direct_answer' cho chào hỏi hoặc kiến thức phổ thông."
        ),
    )


class SearchFilter(BaseModel):
    """Bộ lọc Metadata tự động trích xuất từ câu hỏi người dùng."""

    source_file: Optional[Literal["rag_architecture.md", "product_specs.md"]] = Field(
        default=None,
        description="Tên file tài liệu cụ thể nếu câu hỏi chỉ đích danh kiến trúc RAG hay thông số sản phẩm.",
    )


class GradeDocuments(BaseModel):
    """Đánh giá mức độ liên quan từng phần của 1 chunk đối với câu hỏi (kể cả câu hỏi ghép)."""

    matched_aspect: str = Field(
        ...,
        description=(
            "Ghi ngắn gọn vế câu hỏi hoặc từ khóa kỹ thuật mà tài liệu trả lời được "
            "(ví dụ: 'mô hình reranker FlashRank'). Ghi 'Không có' nếu hoàn toàn lệch chủ đề."
        ),
    )
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description=(
            "Trả về 'yes' nếu tài liệu trả lời được ÍT NHẤT MỘT VẾ của câu hỏi; "
            "chỉ trả về 'no' khi tài liệu hoàn toàn thuộc chủ đề hoặc sản phẩm khác."
        ),
    )


class GradeHallucinations(BaseModel):
    """Kiểm tra tính trung thực của câu trả lời so với tài liệu gốc (Self-RAG)."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Trả về 'yes' nếu câu trả lời bám sát bằng chứng tài liệu, 'no' nếu có dấu hiệu bịa đặt.",
    )


class GraphState(TypedDict):
    """Trạng thái chia sẻ xuyên suốt giữa các Node trong LangGraph."""

    question: str
    rewritten_query: str
    datasource: str
    metadata_filter: Optional[Dict[str, Any]]
    documents: List[Document]
    formatted_context: str
    context_telemetry: Dict[str, Any]
    citations: List[Dict[str, Any]]
    mcp_result: str
    pending_order: Optional[Dict[str, Any]]
    requires_approval: bool
    approval_status: Literal["none", "pending", "approved", "rejected"]
    generation: str
    hallucination_grade: str
    retry_count: int
