"""Bộ kiểm thử đơn vị Offline cho Module Ingestion & Auto-Domain Registry (WP4).

Không phụ thuộc Ollama hay server ngoại vi, sử dụng FakeVectorStore và tmp_path.
"""

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.documents import Document

from src.rag.hybrid_retriever import (
    _infer_query_domain,
    clear_domain_registry,
    register_domain_keywords,
)
from src.rag.indexing import (
    compute_sha256,
    extract_text_from_pdf,
    ingest_file,
    list_indexed_documents,
    load_and_split_documents,
    load_manifest,
    split_single_document,
    upsert_documents,
)


class FakeVectorStore:
    """Mock VectorStore nhẹ phục vụ kiểm thử offline mô phỏng ChromaDB."""

    def __init__(self):
        self.documents: List[Document] = []
        self.deleted_ids: List[str] = []

    def add_documents(self, docs: List[Document]) -> List[str]:
        start = len(self.documents)
        self.documents.extend(docs)
        return [f"id_{start + i}" for i in range(len(docs))]

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        matched_ids = []
        matched_docs = []
        matched_metas = []
        for idx, doc in enumerate(self.documents):
            if where:
                match = all(doc.metadata.get(k) == v for k, v in where.items())
                if match:
                    matched_ids.append(f"id_{idx}")
                    matched_docs.append(doc.page_content)
                    matched_metas.append(doc.metadata)
            else:
                matched_ids.append(f"id_{idx}")
                matched_docs.append(doc.page_content)
                matched_metas.append(doc.metadata)

        if limit is not None:
            matched_ids = matched_ids[:limit]
            matched_docs = matched_docs[:limit]
            matched_metas = matched_metas[:limit]

        return {
            "ids": matched_ids,
            "documents": matched_docs,
            "metadatas": matched_metas,
        }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if ids:
            id_set = set(ids)
            surviving = []
            for idx, doc in enumerate(self.documents):
                doc_id = f"id_{idx}"
                if doc_id in id_set:
                    self.deleted_ids.append(doc_id)
                else:
                    surviving.append(doc)
            self.documents = surviving
        elif where:
            surviving = []
            for idx, doc in enumerate(self.documents):
                match = all(doc.metadata.get(k) == v for k, v in where.items())
                if match:
                    self.deleted_ids.append(f"id_{idx}")
                else:
                    surviving.append(doc)
            self.documents = surviving


@pytest.fixture(autouse=True)
def clean_registry():
    """Tự động dọn dẹp registry trước và sau mỗi test."""
    clear_domain_registry()
    yield
    clear_domain_registry()


# =====================================================================
# 1. TEST CHUNKING VÀ METADATA DOC_DOMAIN CHO .MD VÀ .TXT
# =====================================================================
def test_md_chunking_and_metadata_assignment(tmp_path: Path):
    """Kiểm tra tách file .md với Markdown Headers và gắn doc_domain chuẩn xác."""
    md_content = (
        "# Kiến trúc Hệ thống\n\n"
        "Hệ thống Agentic RAG vận hành với mô hình Local và LangGraph.\n\n"
        "## Bộ lập chỉ mục\n\n"
        "Indexing module phân tách tài liệu thành các chunks nhỏ.\n\n"
        "### Tối ưu hóa\n\n"
        "Sử dụng Cosine distance trong ChromaDB để tính độ tương đồng.\n"
    )
    md_file = tmp_path / "architecture.md"
    md_file.write_text(md_content, encoding="utf-8")

    # Ingest với domain tùy chỉnh
    fake_store = FakeVectorStore()
    manifest_path = tmp_path / "manifest.json"

    res = ingest_file(
        file_path=md_file,
        doc_domain="custom_infra",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )

    assert res["status"] == "indexed"
    assert res["chunk_count"] > 0
    assert len(fake_store.documents) == res["chunk_count"]

    for doc in fake_store.documents:
        assert doc.metadata["source"] == "architecture.md"
        assert doc.metadata["doc_domain"] == "custom_infra"
        assert "chunk_index" in doc.metadata

    # Kiểm tra ngữ cảnh Header được đính kèm vào nội dung
    chunk_contents = [d.page_content for d in fake_store.documents]
    assert any("[Mục: Kiến trúc Hệ thống" in c for c in chunk_contents)


def test_txt_chunking_and_metadata_assignment(tmp_path: Path):
    """Kiểm tra tách file .txt thuần túy và gắn doc_domain metadata."""
    txt_content = (
        "Quy trình xử lý đơn hàng nội bộ của doanh nghiệp.\n\n"
        "Bước 1: Tiếp nhận yêu cầu mua hàng từ khách hàng và kiểm tra mã sản phẩm tồn kho qua MCP ERP.\n\n"
        "Bước 2: Áp dụng chiết khấu dựa theo số lượng đặt hàng.\n\n"
        "Bước 3: Yêu cầu phê duyệt con người (HITL) nếu đơn hàng vượt giá trị ngưỡng an toàn.\n"
    )
    txt_file = tmp_path / "erp_process.txt"
    txt_file.write_text(txt_content, encoding="utf-8")

    fake_store = FakeVectorStore()
    manifest_path = tmp_path / "manifest.json"

    res = ingest_file(
        file_path=txt_file,
        doc_domain="erp_workflow",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )

    assert res["status"] == "indexed"
    assert res["chunk_count"] >= 1
    assert len(fake_store.documents) == res["chunk_count"]

    for doc in fake_store.documents:
        assert doc.metadata["source"] == "erp_process.txt"
        assert doc.metadata["doc_domain"] == "erp_workflow"


def test_default_domain_resolution_when_unspecified(tmp_path: Path):
    """Kiểm tra cơ chế fallback tên file khi doc_domain=None."""
    # product_specs -> product_catalog
    prod_file = tmp_path / "product_specs_server.md"
    prod_file.write_text("# Máy chủ Server AI\nGiá bán 100M VND.", encoding="utf-8")
    chunks_prod = split_single_document(prod_file)
    assert chunks_prod[0].metadata["doc_domain"] == "product_catalog"

    # khác -> system_architecture
    arch_file = tmp_path / "rag_pipeline.md"
    arch_file.write_text("# RAG Pipeline\nSử dụng Cross-Encoder.", encoding="utf-8")
    chunks_arch = split_single_document(arch_file)
    assert chunks_arch[0].metadata["doc_domain"] == "system_architecture"


# =====================================================================
# 2. TEST INCREMENTAL INGESTION DỰA TRÊN MÃ BĂM SHA-256
# =====================================================================
def test_sha256_incremental_ingestion_skip_and_reindex(tmp_path: Path):
    """Kiểm tra: Bỏ qua khi file không đổi, và re-index khi nội dung thay đổi."""
    test_file = tmp_path / "service_guide.md"
    test_file.write_text("# Hướng dẫn vận hành v1\nNội dung ban đầu.", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    fake_store = FakeVectorStore()

    # Lần 1: Chưa có trong manifest -> indexed
    res1 = ingest_file(
        file_path=test_file,
        doc_domain="operations",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    assert res1["status"] == "indexed"
    first_hash = res1["sha256"]
    assert first_hash == compute_sha256(test_file)
    assert len(fake_store.documents) >= 1
    first_doc_count = len(fake_store.documents)

    # Lần 2: Gọi lại khi file không đổi -> skipped
    res2 = ingest_file(
        file_path=test_file,
        doc_domain="operations",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    assert res2["status"] == "skipped"
    assert res2["sha256"] == first_hash
    # Số tài liệu trong vector store không bị nạp trùng lặp
    assert len(fake_store.documents) == first_doc_count

    # Lần 3: Thay đổi nội dung tệp tin -> re-index
    test_file.write_text(
        "# Hướng dẫn vận hành v2\nNội dung đã được cập nhật phiên bản mới với nhiều chi tiết.",
        encoding="utf-8",
    )
    res3 = ingest_file(
        file_path=test_file,
        doc_domain="operations",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    assert res3["status"] == "indexed"
    assert res3["sha256"] != first_hash
    assert res3["sha256"] == compute_sha256(test_file)

    # Kiểm tra manifest cập nhật hash mới
    current_manifest = load_manifest(manifest_path)
    assert current_manifest["service_guide.md"]["sha256"] == res3["sha256"]


def test_force_reindex_overrides_hash_check(tmp_path: Path):
    """Kiểm tra cờ force=True bắt buộc nạp lại dù hash không đổi."""
    test_file = tmp_path / "static_doc.txt"
    test_file.write_text("Dữ liệu tĩnh không đổi.", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    fake_store = FakeVectorStore()

    res1 = ingest_file(test_file, vectorstore=fake_store, manifest_path=manifest_path)
    assert res1["status"] == "indexed"

    res_force = ingest_file(
        test_file, vectorstore=fake_store, manifest_path=manifest_path, force=True
    )
    assert res_force["status"] == "indexed"


# =====================================================================
# 3. TEST LIST_INDEXED_DOCUMENTS
# =====================================================================
def test_list_indexed_documents_returns_accurate_metadata(tmp_path: Path):
    """Kiểm tra hàm list_indexed_documents trả về danh sách chi tiết các trường yêu cầu."""
    manifest_path = tmp_path / "manifest.json"
    fake_store = FakeVectorStore()

    file_a = tmp_path / "alpha.md"
    file_a.write_text("# Alpha\nNội dung tài liệu Alpha.", encoding="utf-8")

    file_b = tmp_path / "beta.txt"
    file_b.write_text("Nội dung tài liệu Beta dành cho kinh doanh.", encoding="utf-8")

    ingest_file(
        file_a,
        doc_domain="domain_alpha",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    ingest_file(
        file_b,
        doc_domain="domain_beta",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )

    indexed = list_indexed_documents(manifest_path=manifest_path)
    assert len(indexed) == 2

    # Kiểm tra đủ 5 trường bắt buộc
    for item in indexed:
        assert "filename" in item
        assert "doc_domain" in item
        assert "sha256" in item
        assert "chunk_count" in item
        assert "updated_at" in item

    docs_by_name = {item["filename"]: item for item in indexed}
    assert "alpha.md" in docs_by_name
    assert docs_by_name["alpha.md"]["doc_domain"] == "domain_alpha"
    assert docs_by_name["alpha.md"]["chunk_count"] >= 1
    assert len(docs_by_name["alpha.md"]["sha256"]) == 64

    assert "beta.txt" in docs_by_name
    assert docs_by_name["beta.txt"]["doc_domain"] == "domain_beta"
    assert docs_by_name["beta.txt"]["chunk_count"] >= 1


# =====================================================================
# 4. TEST REGISTER_DOMAIN_KEYWORDS VÀ DYNAMIC ROUTING
# =====================================================================
def test_register_domain_keywords_and_infer_query_domain():
    """Kiểm tra đăng ký dynamic domain keywords và định tuyến câu hỏi."""
    # Mặc định hai domain cốt lõi
    assert (
        _infer_query_domain("Kiến trúc hệ thống RAG và mô hình triển khai LangGraph")
        == "system_architecture"
    )
    assert (
        _infer_query_domain("Kiểm tra giá bán và chính sách bảo hành ổ cứng SSD")
        == "product_catalog"
    )
    assert _infer_query_domain("Hôm nay thời tiết thế nào?") is None

    # Đăng ký domain động mới: Nhân sự & Pháp lý
    register_domain_keywords(
        domain="human_resources",
        keywords=["nhân sự", "nghỉ phép", "bảo hiểm xã hội", "hợp đồng lao động"],
    )

    assert (
        _infer_query_domain("Quy trình xin nghỉ phép và chế độ bảo hiểm xã hội thế nào?")
        == "human_resources"
    )

    # Đăng ký thêm domain động: Tài chính & Kế toán
    register_domain_keywords(
        domain="finance_accounting",
        keywords=("hóa đơn vat", "thuế thu nhập", "thanh toán hóa đơn"),
    )

    assert (
        _infer_query_domain("Cách thức xuất hóa đơn VAT cho khách hàng doanh nghiệp")
        == "finance_accounting"
    )

    # Câu hỏi kết hợp không rõ ràng -> trả về None
    assert (
        _infer_query_domain("So sánh giá bán sản phẩm với kiến trúc hệ thống RAG")
        is None
    )

    # Dọn dẹp registry
    clear_domain_registry()
    assert (
        _infer_query_domain("Quy trình xin nghỉ phép và chế độ bảo hiểm xã hội thế nào?")
        is None
    )


# =====================================================================
# 5. TEST XỬ LÝ ĐỊNH DẠNG PDF VÀ LAZY IMPORT PYPDF
# =====================================================================
def test_pdf_lazy_import_raises_clear_error_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Khi pypdf không được cài đặt, việc đọc file PDF phải bắn ra RuntimeError có thông báo rõ ràng."""
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    # Giả lập pypdf không tồn tại trong môi trường
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(RuntimeError) as exc_info:
        extract_text_from_pdf(pdf_file)

    assert "pypdf" in str(exc_info.value).lower()

    with pytest.raises(RuntimeError) as exc_info2:
        split_single_document(pdf_file)

    assert "pypdf" in str(exc_info2.value).lower()


def test_pdf_extraction_with_mock_pypdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Kiểm tra đọc và chunking PDF khi có thư viện pypdf (qua mock pypdf)."""
    pdf_file = tmp_path / "whitepaper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    class MockPage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class MockPdfReader:
        def __init__(self, stream: Any):
            self.pages = [
                MockPage("Trang 1: Báo cáo kỹ thuật AI Agent và Multi-Agent Systems."),
                MockPage("Trang 2: Đánh giá hiệu năng và chi phí truy vấn vector."),
            ]

    mock_pypdf = types.ModuleType("pypdf")
    mock_pypdf.PdfReader = MockPdfReader  # type: ignore
    monkeypatch.setitem(sys.modules, "pypdf", mock_pypdf)

    fake_store = FakeVectorStore()
    manifest_path = tmp_path / "manifest.json"

    res = ingest_file(
        file_path=pdf_file,
        doc_domain="ai_research",
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )

    assert res["status"] == "indexed"
    assert res["chunk_count"] >= 1
    assert len(fake_store.documents) == res["chunk_count"]
    assert fake_store.documents[0].metadata["doc_domain"] == "ai_research"
    assert "AI Agent" in fake_store.documents[0].page_content


# =====================================================================
# 6. TEST UPSERT_DOCUMENTS HÀNG LOẠT VÀ LOAD_AND_SPLIT_DOCUMENTS
# =====================================================================
def test_upsert_documents_batch_and_load_split(tmp_path: Path):
    """Kiểm tra quét thư mục nạp hàng loạt và load_and_split_documents."""
    f1 = tmp_path / "doc1.md"
    f1.write_text("# Doc 1\nNội dung tài liệu thứ nhất.", encoding="utf-8")

    f2 = tmp_path / "doc2.txt"
    f2.write_text("Nội dung tài liệu thứ hai.", encoding="utf-8")

    fake_store = FakeVectorStore()
    manifest_path = tmp_path / "manifest.json"

    # Nạp toàn bộ thư mục
    results = upsert_documents(
        data_dir=tmp_path,
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    assert len(results) == 2
    assert all(r["status"] == "indexed" for r in results)

    # Nạp lại lần 2 -> skipped cả 2
    results_second = upsert_documents(
        data_dir=tmp_path,
        vectorstore=fake_store,
        manifest_path=manifest_path,
    )
    assert len(results_second) == 2
    assert all(r["status"] == "skipped" for r in results_second)

    # load_and_split_documents đọc đủ cả 2 file
    docs = load_and_split_documents(data_dir=tmp_path)
    assert len(docs) >= 2


def test_unsupported_file_extension_raises_error(tmp_path: Path):
    """Kiểm tra ném lỗi ValueError đối với định dạng không được hỗ trợ."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name\n1,Alpha", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        split_single_document(csv_file)

    assert "không được hỗ trợ" in str(exc.value)
