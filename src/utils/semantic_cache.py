"""Semantic Cache (ChromaDB cosine) cho câu trả lời RAG: giảm độ trễ cho câu hỏi tương tự.

- Ngưỡng tương đồng cosine mặc định >= 0.92, TTL mặc định 3600 giây.
- `embed_fn` có thể inject (test offline, không cần Ollama); nếu không, lazy-load OllamaEmbeddings.
- `should_bypass_cache` luôn bỏ qua cache với truy vấn ERP/ghi/tồn kho/đơn hàng hoặc chứa PII.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600

# Từ khóa ERP / ghi / tồn kho / đơn hàng -> luôn bypass cache (dữ liệu thời gian thực hoặc có side-effect).
ERP_BYPASS_KEYWORDS: tuple[str, ...] = (
    "tồn kho",
    "ton kho",
    "chiết khấu",
    "chiet khau",
    "chốt đơn",
    "chot don",
    "đặt mua",
    "dat mua",
    "đặt hàng",
    "dat hang",
    "đơn hàng",
    "don hang",
    "mua ",
    "trừ kho",
    "tru kho",
    "nhập kho",
    "xuất kho",
    "thanh toán",
    "tính tổng tiền",
    "tính tiền",
    "mcp_tools",
    "mcp",
    "erp",
    "inventory",
    "stock",
    "discount",
    "checkout",
    "purchase",
    "order",
    "sku",
    "approve",
    "phê duyệt",
)
BYPASS_DATASOURCES: frozenset[str] = frozenset({"mcp_tools"})

# Dự phòng khi không import được pii_proxy: nhận diện nhanh email / số điện thoại / CCCD.
_PII_FALLBACK_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+"  # email
    r"|(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)"  # SĐT VN
    r"|(?<!\d)\d{9}(?!\d)|(?<!\d)\d{12}(?!\d)"  # CMND / CCCD
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().lower()


def contains_pii(query: str) -> bool:
    """True nếu truy vấn chứa PII hoặc prompt-injection (không bao giờ được cache)."""
    try:
        from src.utils.pii_proxy import pii_proxy

        scan = pii_proxy.mask(query)
        return bool(scan.has_pii or scan.is_injection_blocked)
    except Exception:  # pragma: no cover - fallback an toàn
        return bool(_PII_FALLBACK_RE.search(query))


def should_bypass_cache(query: str, datasource: str | None = None) -> bool:
    """Luôn bỏ qua cache với truy vấn ERP/ghi/tồn kho/đơn hàng hoặc chứa PII."""
    if datasource in BYPASS_DATASOURCES:
        return True
    normalized = _normalize(query)
    if any(kw in normalized for kw in ERP_BYPASS_KEYWORDS):
        return True
    return contains_pii(query)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class CacheHit(Dict[str, Any]):
    """Kết quả cache: {'response', 'citations', 'similarity', 'query'}."""


class SemanticCache:
    """Cache ngữ nghĩa dùng ChromaDB (cosine) với TTL; có thể inject embed_fn để test offline."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "semantic_cache",
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        time_fn: Callable[[], float] = time.time,
    ):
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._embed_fn = embed_fn
        self._time_fn = time_fn
        self._collection: Any = None
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._evictions = 0

    # ------------------------------------------------------------------ internals
    def _embed(self, text: str) -> List[float]:
        if self._embed_fn is None:
            from src.rag.indexing import get_embeddings  # lazy: chỉ cần Ollama khi dùng thật

            embedder = get_embeddings()
            self._embed_fn = embedder.embed_query
        return [float(x) for x in self._embed_fn(text)]

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb  # lazy import

                if self.persist_directory is not None:
                    self.persist_directory.mkdir(parents=True, exist_ok=True)
                    client = chromadb.PersistentClient(path=str(self.persist_directory))
                else:
                    client = chromadb.EphemeralClient()
                    # EphemeralClient dùng chung 1 hệ thống trong process -> tên collection duy nhất
                    self.collection_name = f"{self.collection_name}_{uuid.uuid4().hex[:8]}"
                self._collection = client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                self._collection = _InMemoryCosineCollection()
        return self._collection

    def _evict_expired(self) -> int:
        collection = self._get_collection()
        now = self._time_fn()
        data = collection.get(include=["metadatas"])
        expired = [
            cid
            for cid, meta in zip(data.get("ids", []), data.get("metadatas", []))
            if meta and float(meta.get("expires_at", 0)) <= now
        ]
        if expired:
            collection.delete(ids=expired)
            self._evictions += len(expired)
        return len(expired)

    # ------------------------------------------------------------------ public API
    def get(self, query: str) -> Optional[CacheHit]:
        """Trả về CacheHit nếu có câu hỏi tương tự (cosine >= ngưỡng) chưa hết hạn, ngược lại None."""
        with self._lock:
            collection = self._get_collection()
            if collection.count() == 0:
                self._misses += 1
                return None
            self._evict_expired()
            if collection.count() == 0:
                self._misses += 1
                return None

            result = collection.query(
                query_embeddings=[self._embed(query)],
                n_results=1,
                include=["metadatas", "distances"],
            )
            ids = result.get("ids", [[]])[0]
            if not ids:
                self._misses += 1
                return None
            distance = float(result["distances"][0][0])
            similarity = 1.0 - distance  # cosine distance -> similarity
            meta = result["metadatas"][0][0] or {}
            if similarity < self.similarity_threshold:
                self._misses += 1
                return None

            self._hits += 1
            return CacheHit(
                response=meta.get("response", ""),
                citations=json.loads(meta.get("citations", "[]")),
                similarity=round(similarity, 4),
                query=meta.get("query", ""),
            )

    def set(self, query: str, response: str, citations: List[Any] | None = None) -> None:
        """Lưu cặp (câu hỏi, câu trả lời, trích dẫn) với TTL."""
        with self._lock:
            collection = self._get_collection()
            now = self._time_fn()
            collection.upsert(
                ids=[hashlib.sha1(_normalize(query).encode("utf-8")).hexdigest()],
                embeddings=[self._embed(query)],
                documents=[query],
                metadatas=[
                    {
                        "query": query,
                        "response": response,
                        "citations": json.dumps(citations or [], ensure_ascii=False, default=str),
                        "created_at": now,
                        "expires_at": now + self.ttl_seconds,
                    }
                ],
            )

    def record_bypass(self) -> None:
        self._bypasses += 1

    def clear(self) -> int:
        """Xóa toàn bộ cache, trả về số entry đã xóa."""
        with self._lock:
            collection = self._get_collection()
            ids = collection.get(include=[]).get("ids", [])
            if ids:
                collection.delete(ids=ids)
            return len(ids)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            lookups = self._hits + self._misses
            try:
                size = self._get_collection().count()
            except Exception:
                size = 0
            return {
                "entries": size,
                "hits": self._hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / lookups, 4) if lookups else 0.0,
                "ttl_seconds": self.ttl_seconds,
                "similarity_threshold": self.similarity_threshold,
            }


class _InMemoryCosineCollection:
    """Fallback nhẹ thuần Python khi môi trường CI không cài chromadb."""

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def count(self) -> int:
        return len(self._items)

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._items[cid] = {
                "embedding": [float(x) for x in emb],
                "document": doc,
                "metadata": dict(meta),
            }

    upsert = add

    def get(self, include: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        ids = list(self._items.keys())
        return {
            "ids": ids,
            "documents": [self._items[i]["document"] for i in ids],
            "metadatas": [self._items[i]["metadata"] for i in ids],
        }

    def delete(self, ids: List[str]) -> None:
        for cid in ids:
            self._items.pop(cid, None)

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        sim = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
        return 1.0 - max(min(sim, 1.0), -1.0)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 1,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[List[Any]]]:
        q_emb = query_embeddings[0]
        scored = [
            (self._cosine_distance(q_emb, item["embedding"]), cid, item)
            for cid, item in self._items.items()
        ]
        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "ids": [[x[1] for x in top]],
            "distances": [[x[0] for x in top]],
            "documents": [[x[2]["document"] for x in top]],
            "metadatas": [[x[2]["metadata"] for x in top]],
        }
