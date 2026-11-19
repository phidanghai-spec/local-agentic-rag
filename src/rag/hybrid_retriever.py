"""Module truy xuất lai (Hybrid Search: BM25 + Chroma Cosine) kèm FlashRank Reranker."""

import math
import re
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


_DYNAMIC_DOMAIN_REGISTRY: Dict[str, set[str]] = {}


def register_domain_keywords(
    domain: str, keywords: tuple[str, ...] | list[str]
) -> None:
    """Đăng ký danh sách từ khóa đại diện cho một domain động mới."""
    if not domain:
        return
    cleaned_keywords = {k.strip().lower() for k in keywords if k and k.strip()}
    if domain in _DYNAMIC_DOMAIN_REGISTRY:
        _DYNAMIC_DOMAIN_REGISTRY[domain].update(cleaned_keywords)
    else:
        _DYNAMIC_DOMAIN_REGISTRY[domain] = set(cleaned_keywords)


def clear_domain_registry() -> None:
    """Xóa danh sách domain đã đăng ký động (hỗ trợ kiểm thử cô lập)."""
    _DYNAMIC_DOMAIN_REGISTRY.clear()


def get_registered_domains() -> Dict[str, List[str]]:
    """Trả về từ điển các domain và từ khóa tương ứng đã được đăng ký."""
    return {k: sorted(list(v)) for k, v in _DYNAMIC_DOMAIN_REGISTRY.items()}


def _infer_query_domain(query: str) -> Optional[str]:
    """Nhận diện nhanh ý định câu hỏi thuộc về kiến trúc hệ thống, thông số sản phẩm hoặc domain động."""
    q = (query or "").lower()
    if not q:
        return None

    arch_keywords = (
        "kiến trúc", "rag", "rerank", "flashrank", "ollama", "langgraph",
        "triển khai", "hệ thống", "mô hình", "bm25", "chromadb", "embedding", "crag",
    )
    prod_keywords = (
        "giá bán", "giá niêm yết", "tồn kho", "ai-ws-x1", "ssd-ent-4t",
        "bảo hành", "đổi trả", "ổ cứng", "mua", "chiết khấu", "mã sản phẩm",
    )

    matched_domains: set[str] = set()

    if any(k in q for k in arch_keywords):
        matched_domains.add("system_architecture")

    if any(k in q for k in prod_keywords):
        matched_domains.add("product_catalog")

    for dyn_domain, dyn_keywords in _DYNAMIC_DOMAIN_REGISTRY.items():
        if any(k in q for k in dyn_keywords):
            matched_domains.add(dyn_domain)

    if len(matched_domains) == 1:
        return next(iter(matched_domains))

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


# =====================================================================
# COMPARISON QUERY DECOMPOSITION + RECIPROCAL RANK FUSION (RRF)
# =====================================================================
_COMPARISON_TRIGGER_RE = re.compile(
    r"\bso\s+sánh\b|\bđối\s+chiếu\b|\bkhác\s+(?:nhau|biệt|gì)\b|\bkhác\s+.+?\s+(?:so\s+với|với)\b"
    r"|\bvs\.?(?=\s)|\bversus\b",
    re.IGNORECASE,
)
_ENTITY_SPLIT_RE = re.compile(r"\s*,\s*|\s+(?:và|với|vs\.?|versus|hay|hoặc)\s+", re.IGNORECASE)
_LEADING_NOISE_RE = re.compile(
    r"^(?:hãy\s+|vui\s+lòng\s+|giúp\s+(?:tôi|mình)\s+)?"
    r"(?:so\s+sánh|đối\s+chiếu|điểm\s+khác\s+(?:nhau|biệt)|sự\s+khác\s+(?:nhau|biệt)|khác\s+(?:nhau|biệt))\s*",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(
    r"(?:\s*(?:như\s+thế\s+nào|thế\s+nào|ra\s+sao|là\s+gì|có\s+gì\s+khác(?:\s+nhau)?|khác\s+(?:nhau|biệt)"
    r"(?:\s+(?:như\s+thế\s+nào|ra\s+sao|gì))?|nhé|ạ|nào\s+tốt\s+hơn|cái\s+nào\s+hơn))+\s*[?.!]*\s*$|\s*[?.!]+\s*$",
    re.IGNORECASE,
)


def _clean_fragment(text: str) -> str:
    """Bỏ từ đệm đầu/cuối và khoảng trắng thừa trong một vế của câu so sánh."""
    cleaned = _LEADING_NOISE_RE.sub("", text.strip())
    cleaned = _TRAILING_NOISE_RE.sub("", cleaned)
    cleaned = re.sub(r"^(?:về|giữa|của)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:về|của)$", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" ,;:-")


def _split_entities(text: str) -> List[str]:
    """Tách cụm thực thể 'A, B và C' thành danh sách ['A', 'B', 'C']."""
    parts = [_clean_fragment(p) for p in _ENTITY_SPLIT_RE.split(_clean_fragment(text))]
    return [p for p in parts if p]


def _compose(aspect: str, entity: str) -> str:
    return f"{aspect} {entity}".strip() if aspect else entity


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _decompose_comparison_query(query: str) -> List[str]:
    """
    Phân rã câu hỏi so sánh thành các truy vấn con, mỗi truy vấn tập trung vào MỘT thực thể,
    để mỗi thực thể có suất truy xuất công bằng thay vì một bên chiếm trọn top-k.

    Hỗ trợ các mẫu câu (không phân biệt hoa thường):
    - "So sánh <khía cạnh> giữa A và B"           -> ["<khía cạnh> A", "<khía cạnh> B"]
    - "Điểm khác nhau giữa A và B về <khía cạnh>" -> ["<khía cạnh> A", "<khía cạnh> B"]
    - "So sánh A và/với B về <khía cạnh>"         -> ["A <khía cạnh>", "B <khía cạnh>"]
    - "So sánh A với B" / "A vs B" / "A khác B như thế nào" -> ["A", "B"]

    Trả về `[]` nếu không phải câu hỏi so sánh hoặc không tách được >= 2 thực thể.
    """
    q = (query or "").strip()
    if not q or not _COMPARISON_TRIGGER_RE.search(q):
        return []

    aspect = ""
    entities: List[str] = []

    between = re.search(r"\bgiữa\s+(?P<ents>.+)$", q, flags=re.IGNORECASE)
    if between:
        prefix = q[: between.start()]
        ents_text = between.group("ents")
        tail_aspect = re.search(r"\s+(?:về|ở|theo|trên\s+phương\s+diện)\s+(?P<asp>.+)$", ents_text, re.IGNORECASE)
        if tail_aspect:
            aspect = _clean_fragment(tail_aspect.group("asp"))
            ents_text = ents_text[: tail_aspect.start()]
        if not aspect:
            aspect = _clean_fragment(prefix)
        entities = _split_entities(ents_text)
    else:
        body = _clean_fragment(q)
        about = re.search(r"\s+(?:về|ở|theo|trên\s+phương\s+diện)\s+(?P<asp>.+)$", body, re.IGNORECASE)
        if about:
            aspect = _clean_fragment(about.group("asp"))
            body = body[: about.start()]
        diff = re.match(r"^(?P<a>.+?)\s+khác(?:\s+(?:gì|biệt|nhau))?\s+(?:so\s+với|với)\s+(?P<b>.+)$", body, re.IGNORECASE)
        if diff:
            entities = [_clean_fragment(diff.group("a")), _clean_fragment(diff.group("b"))]
        else:
            entities = _split_entities(body)
        # Với mẫu "A và B về X", đặt thực thể trước khía cạnh để giữ trật tự tự nhiên.
        if aspect and len(entities) >= 2:
            return _dedupe_keep_order([f"{e} {aspect}" for e in entities if e])

    entities = _dedupe_keep_order([e for e in entities if e])
    if len(entities) < 2:
        return []
    return _dedupe_keep_order([_compose(aspect, e) for e in entities])


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}::{doc.page_content}"


def _rrf_merge(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d) + 1).
    Hạng 1 của MỌI danh sách con có điểm bằng nhau nên các thực thể được xen kẽ công bằng;
    chunk xuất hiện ở nhiều danh sách được cộng dồn điểm và đẩy lên trước. Trùng lặp được gộp.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, tuple] = {}
    docs_by_key: Dict[str, Document] = {}

    for list_idx, docs in enumerate(ranked_lists):
        for rank, doc in enumerate(docs or []):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in docs_by_key:
                docs_by_key[key] = doc
                first_seen[key] = (rank, list_idx)

    ordered_keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    merged: List[Document] = []
    for key in ordered_keys:
        doc = docs_by_key[key]
        merged.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rrf_score": round(scores[key], 6)},
            )
        )

    return merged[:top_n] if top_n else merged


async def _retrieve_single_query(
    store: Any,
    compression_retriever: Any,
    query: str,
    metadata_filter: Optional[Dict[str, Any]],
    max_chunks: int = 3,
) -> Optional[List[Document]]:
    """Chạy trọn pipeline Ensemble -> Pre-filter domain -> FlashRank -> Logit/Jaccard cho 1 truy vấn."""
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

    precision_docs = filter_reranked_documents(reranked_docs, query=query, max_chunks=max_chunks)
    if precision_docs:
        return precision_docs

    if reranked_docs:
        return [reranked_docs[0]]
    return None  # FlashRank không trả kết quả -> để caller fallback sang ChromaDB thuần


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

    Câu hỏi so sánh ("So sánh ... giữa A và B") được phân rã thành các truy vấn con theo
    từng thực thể, truy xuất độc lập rồi hợp nhất bằng RRF để cả A và B đều có mặt.
    """
    from src.rag.indexing import build_or_load_vectorstore

    store = build_or_load_vectorstore()
    try:
        compression_retriever = create_hybrid_retriever(
            vectorstore=store,
            metadata_filter=metadata_filter,
        )

        sub_queries = _decompose_comparison_query(query)
        if len(sub_queries) >= 2:
            per_entity_docs: List[List[Document]] = []
            for sub_query in sub_queries:
                sub_docs = await _retrieve_single_query(
                    store, compression_retriever, sub_query, metadata_filter, max_chunks=2
                )
                per_entity_docs.append(sub_docs or [])
            merged = _rrf_merge(per_entity_docs, top_n=max(3, 2 * len(sub_queries)))
            if merged:
                print(
                    f"[Retriever] Comparison decomposition {sub_queries} -> "
                    f"{[len(d) for d in per_entity_docs]} chunks, RRF giữ {len(merged)}"
                )
                return merged

        single_docs = await _retrieve_single_query(
            store, compression_retriever, query, metadata_filter
        )
        if single_docs is not None:
            return single_docs
    except Exception as exc:
        print(f"[Retriever Warning] FlashRank gặp sự cố, chuyển về ChromaDB: {exc}")

    raw_docs = await store.as_retriever(search_kwargs={"k": 2}).ainvoke(query)
    return raw_docs[:2]

