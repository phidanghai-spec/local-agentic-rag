"""Unit test offline cho SemanticCache (không cần Ollama): embedding giả lập tất định + tmp_path."""

import hashlib
import math

import pytest

from src.utils.semantic_cache import SemanticCache, should_bypass_cache

DIM = 64


def fake_embed(text: str) -> list[float]:
    """Bag-of-words hashing embedding: câu có từ trùng nhau -> cosine cao, tất định."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(tmp_path, clock):
    return SemanticCache(
        persist_directory=tmp_path / "cache_db",
        collection_name="test_cache",
        ttl_seconds=3600,
        embed_fn=fake_embed,
        time_fn=clock,
    )


CITATIONS = [{"source": "spec.md", "snippet": "FlashRank chạy trên CPU", "relevance_score": 0.9}]


def test_cache_hit_exact_and_near_duplicate(cache):
    question = "hệ thống sử dụng mô hình reranker nào chạy trên cpu hay gpu"
    cache.set(question, "FlashRank chạy trên CPU.", CITATIONS)

    exact = cache.get(question)
    assert exact is not None
    assert exact["response"] == "FlashRank chạy trên CPU."
    assert exact["citations"] == CITATIONS
    assert exact["similarity"] >= 0.92

    # Chỉ thêm 1 từ (cosine ~0.97) -> vẫn HIT
    near = cache.get(question + " vậy")
    assert near is not None and near["similarity"] >= 0.92
    assert cache.get_stats()["hits"] == 2


def test_cache_miss_for_unrelated_query_and_empty_cache(cache):
    assert cache.get("câu hỏi bất kỳ") is None  # cache rỗng

    cache.set("laptop ai workstation x1 bảo hành bao lâu", "24 tháng", [])
    assert cache.get("thời tiết hôm nay ở hà nội thế nào") is None
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert stats["entries"] == 1


def test_ttl_expiry_evicts_entry(cache, clock):
    question = "chính sách bảo hành laptop là gì"
    cache.set(question, "Bảo hành 24 tháng", [])
    assert cache.get(question) is not None

    clock.now += 3601  # vượt TTL
    assert cache.get(question) is None
    stats = cache.get_stats()
    assert stats["entries"] == 0
    assert stats["evictions"] == 1


def test_clear_returns_count(cache):
    cache.set("câu hỏi một", "a", [])
    cache.set("câu hỏi hai khác hẳn", "b", [])
    assert cache.clear() == 2
    assert cache.get_stats()["entries"] == 0


@pytest.mark.parametrize(
    "query",
    [
        "Kiểm tra tồn kho của mã AI-WS-X1",
        "Tính tiền với chiết khấu 10%",
        "Chốt đơn 2 chiếc giúp tôi",
        "Tôi muốn đặt mua laptop",
    ],
)
def test_erp_queries_always_bypass(query):
    assert should_bypass_cache(query) is True


def test_datasource_and_pii_bypass():
    assert should_bypass_cache("Xin chào", datasource="mcp_tools") is True
    assert should_bypass_cache("Liên hệ tôi qua email nguyenvan@example.com") is True
    assert should_bypass_cache("Số điện thoại của tôi là 0912345678") is True


def test_plain_knowledge_query_not_bypassed():
    assert should_bypass_cache("Hệ thống sử dụng mô hình Reranker nào?") is False
    assert should_bypass_cache("Hệ thống sử dụng mô hình Reranker nào?", "vectorstore") is False
