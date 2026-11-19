"""Module lập chỉ mục (Indexing) tài liệu đa định dạng (.md, .txt, .pdf) vào ChromaDB với Cosine Similarity và SHA-256 Incremental Manifest."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import settings

DEFAULT_MANIFEST_PATH = settings.data_dir / "ingestion_manifest.json"

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def compute_sha256(file_path: Path | str) -> str:
    """Tính toán mã băm SHA-256 theo từng khối để kiểm tra thay đổi nội dung tệp."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Optional[Path | str] = None) -> Dict[str, Dict[str, Any]]:
    """Tải tệp manifest JSON ghi nhận trạng thái chỉ mục incremental."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(
    manifest: Dict[str, Dict[str, Any]], manifest_path: Optional[Path | str] = None
) -> None:
    """Lưu tệp manifest JSON ghi nhận trạng thái chỉ mục incremental."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def list_indexed_documents(
    manifest_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Trả về danh sách metadata của các tài liệu đã được lập chỉ mục trong manifest:
    filename, doc_domain, sha256, chunk_count, updated_at.
    """
    manifest = load_manifest(manifest_path)
    result: List[Dict[str, Any]] = []
    for filename, meta in manifest.items():
        result.append(
            {
                "filename": meta.get("filename", filename),
                "doc_domain": meta.get("doc_domain", ""),
                "sha256": meta.get("sha256", ""),
                "chunk_count": meta.get("chunk_count", 0),
                "updated_at": meta.get("updated_at", ""),
            }
        )
    return result


def extract_text_from_pdf(pdf_path: Path | str) -> str:
    """
    Trích xuất toàn bộ văn bản từ file PDF.
    Lazy-import pypdf với thông báo lỗi RuntimeError rõ ràng nếu pypdf chưa được cài đặt.
    """
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Thư viện `pypdf` chưa được cài đặt. Vui lòng cài đặt qua `pip install pypdf` để đọc tệp PDF."
        ) from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {path}")

    reader = pypdf.PdfReader(str(path))
    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def enrich_chunks_with_context(
    chunks: List[Document], default_domain: Optional[str] = None
) -> List[Document]:
    """
    Chỉ gắn đường dẫn tiêu đề mục (h1/h2/h3) riêng của từng chunk vào page_content
    để tránh ô nhiễm từ khóa sang các chunk khác trong cùng 1 file.
    Gắn metadata `doc_domain`: ưu tiên default_domain, fallback product_catalog hoặc system_architecture.
    """
    enriched_chunks: List[Document] = []

    for chunk in chunks:
        source_name = str(chunk.metadata.get("source", "")).split("/")[-1].split("\\")[-1]

        section_headers = [
            chunk.metadata[key]
            for key in ("h1", "h2", "h3")
            if key in chunk.metadata and chunk.metadata[key]
        ]
        section_path = f"[Mục: {' / '.join(section_headers)}]\n" if section_headers else ""

        if default_domain:
            resolved_domain = default_domain
        elif "product_specs" in source_name.lower() or "product" in source_name.lower():
            resolved_domain = "product_catalog"
        else:
            resolved_domain = "system_architecture"

        enriched_doc = Document(
            page_content=f"{section_path}{chunk.page_content.strip()}",
            metadata={
                **chunk.metadata,
                "doc_domain": resolved_domain,
            },
        )
        enriched_chunks.append(enriched_doc)

    return enriched_chunks


def split_single_document(
    file_path: Path | str, doc_domain: Optional[str] = None
) -> List[Document]:
    """Phân tách một tệp tài liệu (.md, .txt, .pdf) thành danh sách Document chunks kèm metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tài liệu: {path}")

    ext = path.suffix.lower()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if ext == ".md":
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        header_docs = md_splitter.split_text(raw_text)
        split_docs = text_splitter.split_documents(header_docs)
        for idx, doc in enumerate(split_docs):
            doc.metadata["source"] = path.name
            doc.metadata["chunk_index"] = idx
        return enrich_chunks_with_context(split_docs, default_domain=doc_domain)

    elif ext == ".txt":
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        split_docs = text_splitter.create_documents([raw_text])
        for idx, doc in enumerate(split_docs):
            doc.metadata["source"] = path.name
            doc.metadata["chunk_index"] = idx
        return enrich_chunks_with_context(split_docs, default_domain=doc_domain)

    elif ext == ".pdf":
        raw_text = extract_text_from_pdf(path)
        split_docs = text_splitter.create_documents([raw_text])
        for idx, doc in enumerate(split_docs):
            doc.metadata["source"] = path.name
            doc.metadata["chunk_index"] = idx
        return enrich_chunks_with_context(split_docs, default_domain=doc_domain)

    else:
        raise ValueError(
            f"Định dạng tệp không được hỗ trợ: {ext}. Hệ thống chỉ hỗ trợ .md, .txt, .pdf."
        )


def get_embeddings() -> Any:
    """Khởi tạo mô hình nhúng Local qua Ollama (lazy-import để tránh nạp chậm lúc import)."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vectorstore() -> Any:
    """Kết nối tới ChromaDB với không gian khoảng cách Cosine (lazy-import để chạy offline)."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.persist_directory),
        collection_metadata={"hnsw:space": settings.chroma_distance_space},
    )


def ingest_file(
    file_path: Path | str,
    doc_domain: Optional[str] = None,
    vectorstore: Optional[Any] = None,
    manifest_path: Optional[Path | str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Lập chỉ mục một file đơn lẻ (.md, .txt, .pdf) vào Vector Store kèm SHA-256 incremental check.
    Nếu nội dung không đổi và force=False, bỏ qua (skipped) để tiết kiệm tài nguyên.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cần lập chỉ mục: {path}")

    file_hash = compute_sha256(path)
    manifest = load_manifest(manifest_path)
    manifest_entry = manifest.get(path.name)

    if not force and manifest_entry and manifest_entry.get("sha256") == file_hash:
        return {
            "filename": path.name,
            "doc_domain": manifest_entry.get("doc_domain", doc_domain or "system_architecture"),
            "sha256": file_hash,
            "chunk_count": manifest_entry.get("chunk_count", 0),
            "updated_at": manifest_entry.get("updated_at", ""),
            "status": "skipped",
        }

    chunks = split_single_document(path, doc_domain=doc_domain)
    resolved_domain = (
        chunks[0].metadata.get("doc_domain")
        if chunks
        else (doc_domain or "system_architecture")
    )

    target_store = vectorstore if vectorstore is not None else get_vectorstore()
    if target_store is not None:
        try:
            if hasattr(target_store, "delete"):
                existing = target_store.get(where={"source": path.name})
                if existing and existing.get("ids"):
                    target_store.delete(ids=existing["ids"])
        except Exception:
            pass
        if chunks:
            target_store.add_documents(chunks)

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "filename": path.name,
        "doc_domain": resolved_domain,
        "sha256": file_hash,
        "chunk_count": len(chunks),
        "updated_at": now_iso,
        "status": "indexed",
    }
    manifest[path.name] = record
    save_manifest(manifest, manifest_path)
    return record


def upsert_documents(
    data_dir: Optional[Path | str] = None,
    files: Optional[List[Path | str]] = None,
    doc_domain: Optional[str] = None,
    vectorstore: Optional[Any] = None,
    manifest_path: Optional[Path | str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """
    Lập chỉ mục hàng loạt tệp tin (.md, .txt, .pdf) theo cơ chế incremental SHA-256.
    """
    target_files: List[Path] = []
    if files:
        target_files = [Path(f) for f in files]
    else:
        dir_path = Path(data_dir) if data_dir else settings.data_dir
        if not dir_path.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {dir_path}")
        supported_exts = {".md", ".txt", ".pdf"}
        target_files = sorted(
            [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        )

    results: List[Dict[str, Any]] = []
    for file_path in target_files:
        res = ingest_file(
            file_path=file_path,
            doc_domain=doc_domain,
            vectorstore=vectorstore,
            manifest_path=manifest_path,
            force=force,
        )
        results.append(res)
    return results


def load_and_split_documents(
    data_dir: Optional[Path | str] = None, doc_domain: Optional[str] = None
) -> List[Document]:
    """Đọc toàn bộ file .md, .txt, .pdf trong thư mục và phân tách thành chunks."""
    target_dir = Path(data_dir) if data_dir else settings.data_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {target_dir}")

    all_chunks: List[Document] = []
    supported_exts = {".md", ".txt", ".pdf"}
    files = sorted(
        [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
    )
    for f in files:
        all_chunks.extend(split_single_document(f, doc_domain=doc_domain))
    return all_chunks


def load_and_split_markdown_files(data_dir: Optional[Path | str] = None) -> List[Document]:
    """Hàm tương thích ngược: Đọc toàn bộ file .md và phân tách thành chunks."""
    target_dir = Path(data_dir) if data_dir else settings.data_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {target_dir}")

    all_chunks: List[Document] = []
    for md_path in sorted(target_dir.glob("*.md")):
        all_chunks.extend(split_single_document(md_path))
    return all_chunks


def build_or_load_vectorstore(
    force_reindex: bool = False,
    vectorstore: Optional[Any] = None,
    manifest_path: Optional[Path | str] = None,
) -> Any:
    """Nạp ChromaDB có sẵn; chỉ nhúng lại khi collection rỗng hoặc force_reindex=True."""
    store = vectorstore or get_vectorstore()
    try:
        existing = store.get(limit=1)
        has_existing_data = bool(existing and existing.get("ids"))
    except Exception:
        has_existing_data = False

    if has_existing_data and not force_reindex:
        return store

    if force_reindex and has_existing_data:
        try:
            all_ids = store.get().get("ids", [])
            if all_ids:
                store.delete(ids=all_ids)
        except Exception:
            pass

    upsert_documents(vectorstore=store, manifest_path=manifest_path, force=force_reindex)
    return store


if __name__ == "__main__":
    store = build_or_load_vectorstore(force_reindex=True)
    count = len(store.get().get("ids", []))
    print(f"Re-index thành công {count} chunks (đã gắn Contextual Headers & doc_domain).")
