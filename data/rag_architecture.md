# Kiến trúc Local Agentic RAG (Chuẩn LangChain & LangGraph)

## 1. Tổng quan Hệ thống
Hệ thống Local Agentic RAG được thiết kế để vận hành hoàn toàn nội bộ (On-premise / Localhost) nhằm bảo vệ dữ liệu nhạy cảm của doanh nghiệp và không phát sinh chi phí API đám mây.

### Thành phần cốt lõi
- **Mô hình Ngôn ngữ (LLM):** `qwen2.5-coder:7b` chạy qua Ollama, nhiệt độ `temperature=0` để đảm bảo tính nhất quán khi định tuyến và chấm điểm tài liệu.
- **Mô hình Nhúng (Embeddings):** `nomic-embed-text` tạo vector 768 chiều.
- **Kho Vector (Vector Store):** ChromaDB sử dụng độ đo **Cosine Similarity** (`hnsw:space: cosine`).

## 2. Cơ chế Truy xuất Lai (Hybrid Search & Reranking)
Để giải quyết nhược điểm tìm kiếm từ khóa chính xác của Vector Search thuần túy, hệ thống kết hợp:
1. **BM25 Retriever (Sparse Retrieval):** Tìm kiếm khớp từ khóa chính xác (mã sản phẩm, tên hàm, thuật ngữ chuyên ngành), lấy top `k=5` với trọng số `0.4`.
2. **Chroma Vector Retriever (Dense Retrieval):** Tìm kiếm theo ngữ nghĩa, lấy top `k=5` với trọng số `0.6`.
3. **FlashRank Reranker:** Sử dụng mô hình Cross-Encoder `ms-marco-MultiBERT-L-12` chạy trên CPU để chấm điểm lại và giữ lại `top_n=3` đoạn văn bản chất lượng nhất.

## 3. Vòng lặp Tự sửa lỗi (Adaptive CRAG & Self-RAG)
- **Logical Routing:** Phân loại câu hỏi của người dùng thành `vectorstore`, `mcp_tools` (tra cứu database/tính toán) hoặc `direct_answer`.
- **CRAG (Corrective RAG):** Chấm điểm từng tài liệu (`yes`/`no`). Nếu không có tài liệu đạt chuẩn, hệ thống tự động viết lại truy vấn (**Query Rewriting**) tối đa 2 lần (`retry_count <= 2`).
- **Self-RAG (Hallucination Check):** Kiểm tra câu trả lời sinh ra có bám sát 100% tài liệu ngữ cảnh hay không trước khi trả về người dùng.
