"""Module truy xuất lai (Hybrid Search: BM25 + Chroma Cosine) kèm FlashRank Reranker."""

import math
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.config import settings


def restore_documents_from_chroma(vectorstore: Any) -> List[Document]:
    """Khôi phục danh sách Document từ ChromaDB để dựng lại BM25 không cần embed lại."""
    stored = vectorstore.get(include=["documents", "metadatas"])
    raw_docs = stored.get("documents") or []
    raw_metas = stored.get("metadatas") or []

    restored: List[Document] = []
    for idx, content in enumerate(raw_docs):
        if not content:
            continue
        metadata = raw_metas[idx] if idx < len(raw_metas) and raw_metas[idx] else {}
        restored.append(Document(page_content=content, metadata=metadata))

    return restored


def _matches_filter(doc: Document, metadata_filter: Optional[Dict[str, Any]]) -> bool:
    """Kiểm tra Document có khớp điều kiện lọc metadata hay không (dành cho BM25)."""
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if doc.metadata.get(key) != expected:
            return False
    return True


def create_hybrid_retriever(
    vectorstore: Optional[Any] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Any:
    """Tạo EnsembleRetriever (BM25 + Chroma Cosine) bọc trong FlashRank Reranker."""
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    from langchain_community.retrievers import BM25Retriever
    from src.rag.indexing import build_or_load_vectorstore

    store = vectorstore or build_or_load_vectorstore()
    all_docs = restore_documents_from_chroma(store)

    filtered_docs = [doc for doc in all_docs if _matches_filter(doc, metadata_filter)]
    bm25_source = filtered_docs if filtered_docs else all_docs

    if not bm25_source:
        raise ValueError("Kho tài liệu ChromaDB đang rỗng. Hãy thêm file .md vào thư mục data/.")

    bm25_retriever = BM25Retriever.from_documents(bm25_source)
    bm25_retriever.k = settings.bm25_k

    vector_search_kwargs: Dict[str, Any] = {"k": settings.vector_k}
    if metadata_filter:
        vector_search_kwargs["filter"] = metadata_filter

    chroma_retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs=vector_search_kwargs,
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=settings.ensemble_weights,
    )

    compressor = FlashrankRerank(
        model=settings.flashrank_model,
        top_n=settings.flashrank_top_n,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Tính độ trùng lặp từ vựng giữa 2 chunk để khử trùng lặp."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a.intersection(words_b))
    union = len(words_a.union(words_b))
    return intersection / union


def _sigmoid_to_logit(score: float) -> float:
    """Chuyển điểm xác suất bão hòa (0.998 - 0.9999) về không gian Logit để tách biên độ rõ hơn."""
    clamped = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.log(clamped / (1.0 - clamped))


def _infer_query_domain(query: str) -> Optional[str]:
    """Nhận diện nhanh ý định câu hỏi thuộc về kiến trúc hệ thống hay thông số sản phẩm bán ra."""
    q = query.lower()
    arch_keywords = (
        "kiến trúc", "rag", "rerank", "flashrank", "ollama", "langgraph",
        "triển khai", "hệ thống", "mô hình", "bm25", "chromadb", "embedding", "crag",
    )
    prod_keywords = (
        "giá bán", "giá niêm yết", "tồn kho", "ai-ws-x1", "ssd-ent-4t",
        "bảo hành", "đổi trả", "ổ cứng", "mua", "chiết khấu", "mã sản phẩm",
    )

    has_arch = any(k in q for k in arch_keywords)
    has_prod = any(k in q for k in prod_keywords)

    if has_arch and not has_prod:
        return "system_architecture"
    if has_prod and not has_arch:
        return "product_catalog"
    return None


def prefilter_by_domain(docs: List[Document], query: str) -> List[Document]:
    """
    Lọc trước theo doc_domain ngay sau EnsembleRetriever và TRƯỚC KHI chạy FlashRank.
    Ngăn tình trạng tài liệu sai domain chiếm mất slot top_n của Cross-Encoder.
    """
    if not docs:
        return []

    target_domain = _infer_query_domain(query)
    if not target_domain:
        return docs

    same_domain_docs = [
        d for d in docs if d.metadata.get("doc_domain") == target_domain
    ]
    return same_domain_docs if same_domain_docs else docs


def filter_reranked_documents(
    docs: List[Document],
    query: str = "",
    min_abs_score: float = 0.25,
    min_logit_margin: float = 4.5,
    max_overlap_ratio: float = 0.75,
    max_chunks: int = 3,
) -> List[Document]:
    """
    Lọc và tái sắp xếp sau FlashRank xử lý triệt để hiện tượng bão hòa Sigmoid:
    1. Điều chỉnh nhẹ điểm Logit nếu tài liệu khớp đúng domain của câu hỏi.
    2. Chuyển điểm về Logit để phân tách các chunk cùng sát ngưỡng 0.999.
    3. Cắt bỏ các chunk có Logit thấp hơn chunk đầu bảng quá `min_logit_margin` và giới hạn `max_chunks`.
    4. Khử trùng lặp bằng Jaccard similarity > `max_overlap_ratio`.
    """
    if not docs:
        return []

    target_domain = _infer_query_domain(query)

    # Tách cứng theo domain nếu nhận diện được phạm vi câu hỏi (có fallback nếu rỗng)
    candidate_docs = docs
    if target_domain:
        same_domain_docs = [
            d for d in docs if d.metadata.get("doc_domain") == target_domain
        ]
        if same_domain_docs:
            candidate_docs = same_domain_docs

    scored_items = []
    for doc in candidate_docs:
        raw_score = float(doc.metadata.get("relevance_score", 0.0))
        logit = _sigmoid_to_logit(raw_score)
        scored_items.append((logit, raw_score, doc))

    scored_items.sort(key=lambda item: item[0], reverse=True)

    top_logit, top_raw_score, top_doc = scored_items[0]
    if top_raw_score < 0.08:
        return []

    filtered_docs: List[Document] = [top_doc]

    for logit, raw_score, doc in scored_items[1:]:
        if len(filtered_docs) >= max_chunks:
            break

        if raw_score < min_abs_score:
            continue

        if (top_logit - logit) > min_logit_margin:
            continue

        is_duplicate = any(
            _jaccard_similarity(doc.page_content, kept.page_content) > max_overlap_ratio
            for kept in filtered_docs
        )
        if not is_duplicate:
            filtered_docs.append(doc)

    return filtered_docs


async def retrieve_with_fallback(
    query: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    min_relevance_score: float = 0.35,
) -> List[Document]:
    """
    Pipeline chuẩn Pre-filter:
    1. EnsembleRetriever (BM25 + Chroma Cosine) lấy tập ứng viên rộng.
    2. Pre-filter theo doc_domain TRƯỚC khi đưa vào FlashRank.
    3. FlashRank Reranker chấm điểm riêng trên các ứng viên đúng domain.
    4. Logit Margin + Jaccard Filter chốt danh sách sạch.
    """
    from src.rag.indexing import build_or_load_vectorstore

    store = build_or_load_vectorstore()
    try:
        compression_retriever = create_hybrid_retriever(
            vectorstore=store,
            metadata_filter=metadata_filter,
        )
        raw_candidates = await compression_retriever.base_retriever.ainvoke(query)

        if metadata_filter and not raw_candidates:
            fallback_retriever = create_hybrid_retriever(
                vectorstore=store,
                metadata_filter=None,
            )
            raw_candidates = await fallback_retriever.base_retriever.ainvoke(query)

        if not raw_candidates:
            return []

        domain_candidates = prefilter_by_domain(raw_candidates, query)
        reranked_docs = list(
            compression_retriever.base_compressor.compress_documents(
                domain_candidates, query
            )
        )

        precision_docs = filter_reranked_documents(reranked_docs, query=query)
        if precision_docs:
            return precision_docs

        if reranked_docs:
            return [reranked_docs[0]]
    except Exception as exc:
        print(f"[Retriever Warning] FlashRank gặp sự cố, chuyển về ChromaDB: {exc}")

    raw_docs = await store.as_retriever(search_kwargs={"k": 2}).ainvoke(query)
    return raw_docs[:2]

