"""Module lập chỉ mục (Indexing) tài liệu Markdown vào ChromaDB với Cosine Similarity."""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import settings


HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def enrich_chunks_with_context(chunks: List[Document]) -> List[Document]:
    """
    Chỉ gắn đường dẫn tiêu đề mục (h1/h2/h3) riêng của từng chunk vào page_content
    để tránh ô nhiễm từ khóa sang các chunk khác trong cùng 1 file.
    Phạm vi file được tách riêng hoàn toàn qua metadata `doc_domain`.
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

        enriched_doc = Document(
            page_content=f"{section_path}{chunk.page_content.strip()}",
            metadata={
                **chunk.metadata,
                "doc_domain": "product_catalog" if "product_specs" in source_name else "system_architecture",
            },
        )
        enriched_chunks.append(enriched_doc)

    return enriched_chunks


def get_embeddings() -> OllamaEmbeddings:
    """Khởi tạo mô hình nhúng Local qua Ollama."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def load_and_split_markdown_files(data_dir: Path | None = None) -> List[Document]:
    """Đọc toàn bộ file .md và phân tách theo cấu trúc tiêu đề + kích thước chunk."""
    target_dir = data_dir or settings.data_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {target_dir}")

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    md_files = sorted(target_dir.glob("*.md"))

    for md_path in md_files:
        raw_text = md_path.read_text(encoding="utf-8")
        header_docs = md_splitter.split_text(raw_text)
        split_docs = text_splitter.split_documents(header_docs)

        for idx, doc in enumerate(split_docs):
            doc.metadata["source"] = md_path.name
            doc.metadata["chunk_index"] = idx
            all_chunks.append(doc)

    return enrich_chunks_with_context(all_chunks)


def get_vectorstore() -> Chroma:
    """Kết nối tới ChromaDB với không gian khoảng cách Cosine."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.persist_directory),
        collection_metadata={"hnsw:space": settings.chroma_distance_space},
    )


def build_or_load_vectorstore(force_reindex: bool = False) -> Chroma:
    """Nạp ChromaDB có sẵn; chỉ nhúng lại khi collection rỗng hoặc force_reindex=True."""
    vectorstore = get_vectorstore()
    existing = vectorstore.get(limit=1)
    has_existing_data = bool(existing and existing.get("ids"))

    if has_existing_data and not force_reindex:
        return vectorstore

    if force_reindex and has_existing_data:
        all_ids = vectorstore.get().get("ids", [])
        if all_ids:
            vectorstore.delete(ids=all_ids)

    chunks = load_and_split_markdown_files()
    if chunks:
        vectorstore.add_documents(chunks)

    return vectorstore


if __name__ == "__main__":
    store = build_or_load_vectorstore(force_reindex=True)
    count = len(store.get().get("ids", []))
    print(f"Re-index thành công {count} chunks (đã gắn Contextual Headers & doc_domain).")
