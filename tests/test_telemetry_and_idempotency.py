"""WP3: Telemetry SQLite (Mini-LangSmith) + commit_erp_order idempotent + list_erp_orders (offline, sandbox DB)."""

import asyncio
import sqlite3

import pytest

from tests.conftest import unwrap_mcp_tool
from src.utils.telemetry import TelemetryStore


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
def test_telemetry_record_and_list(tmp_path):
    store = TelemetryStore(db_path=tmp_path / "telemetry.sqlite")
    store.record_trace(
        thread_id="t1",
        question="Tồn kho AI-WS-X1?",
        datasource="mcp_tools",
        cache_hit=False,
        pii_tags=["EMAIL"],
        hitl_status="pending",
        total_latency_ms=120.5,
        node_latencies={"route_question": 10.0, "call_mcp_tools": 90.256},
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.record_trace(
        thread_id="t2",
        question="x" * 500,
        datasource="semantic_cache",
        cache_hit=True,
        total_latency_ms=5,
        created_at="2026-01-01T00:00:01+00:00",
    )

    traces = store.list_traces(limit=10)
    assert [t["thread_id"] for t in traces] == ["t2", "t1"]  # mới nhất trước
    assert traces[0]["cache_hit"] is True
    assert len(traces[0]["question_preview"]) <= 120
    assert traces[1]["pii_tags"] == ["EMAIL"]
    assert traces[1]["node_latencies"] == {"route_question": 10.0, "call_mcp_tools": 90.26}
    assert traces[1]["hitl_status"] == "pending"
    assert len(store.list_traces(limit=1)) == 1

    # Bền vững: mở lại cùng file vẫn thấy dữ liệu.
    assert len(TelemetryStore(db_path=tmp_path / "telemetry.sqlite").list_traces()) == 2


def test_telemetry_metrics_p50_p95(tmp_path):
    store = TelemetryStore(db_path=tmp_path / "t.sqlite")
    empty = store.get_metrics()
    assert empty["total_requests"] == 0 and empty["cache_hit_rate"] == 0.0
    assert empty["p50_latency_ms"] == 0.0 and empty["hitl_counts"] == {}

    for i in range(1, 101):  # latency 1..100 ms
        store.record_trace(
            thread_id=f"t{i}",
            question="q",
            cache_hit=(i % 4 == 0),  # 25 hit
            hitl_status="approved" if i <= 10 else ("rejected" if i <= 15 else "none"),
            total_latency_ms=float(i),
        )

    m = store.get_metrics()
    assert m["total_requests"] == 100
    assert m["cache_hit_rate"] == 0.25
    assert m["p50_latency_ms"] == pytest.approx(50.5)
    assert m["p95_latency_ms"] == pytest.approx(95.05)
    assert m["hitl_counts"] == {"approved": 10, "rejected": 5, "none": 85}


# ----------------------------------------------------------------------
# ERP idempotency (sandbox DB, KHÔNG đụng data/erp_inventory.db)
# ----------------------------------------------------------------------
@pytest.fixture
def erp_sandbox(tmp_path, monkeypatch):
    from src.agent import nodes

    db = tmp_path / "erp_sandbox.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE inventory (sku TEXT PRIMARY KEY, name TEXT NOT NULL, stock_qty INTEGER NOT NULL, "
            "unit_price_vnd REAL NOT NULL, warehouse TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO inventory VALUES ('AI-WS-X1', 'Laptop X1', 14, 58900000, 'Kho HN')")
        conn.commit()
    monkeypatch.setattr(nodes, "ERP_DB_PATH", db)
    return db


def _stock(db, sku="AI-WS-X1"):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT stock_qty FROM inventory WHERE sku = ?", (sku,)).fetchone()[0]


def _count(db, table):
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_commit_erp_order_idempotent_by_thread_id(erp_sandbox):
    from src.agent.nodes import commit_erp_order

    state = {"pending_order": {"sku": "AI-WS-X1", "quantity": 2, "discount_percent": 10.0}}
    config = {"configurable": {"thread_id": "thread-A"}}

    first = asyncio.run(commit_erp_order(state, config))
    second = asyncio.run(commit_erp_order(state, config))

    assert first["approval_status"] == "approved"
    assert second["approval_status"] == "approved"
    assert "Không trừ kho lần nữa" in second["mcp_result"]
    assert _stock(erp_sandbox) == 12  # chỉ trừ 1 lần (14 - 2)
    assert _count(erp_sandbox, "erp_orders") == 1

    with sqlite3.connect(erp_sandbox) as conn:
        key = conn.execute("SELECT idempotency_key FROM erp_orders").fetchone()[0]
        actions = [r[0] for r in conn.execute("SELECT action FROM erp_audit_log ORDER BY log_id")]
        audit = conn.execute(
            "SELECT thread_id, sku, quantity, discount_percent, total_vnd FROM erp_audit_log ORDER BY log_id LIMIT 1"
        ).fetchone()
    assert key == "thread-A"
    assert actions == ["COMMITTED", "DUPLICATE_IGNORED"]
    assert audit == ("thread-A", "AI-WS-X1", 2, 10.0, round(58900000 * 2 * 0.9, 2))


def test_commit_erp_order_idempotent_by_explicit_key(erp_sandbox):
    from src.agent.nodes import commit_erp_order

    state = {
        "pending_order": {
            "sku": "AI-WS-X1",
            "quantity": 3,
            "discount_percent": 0.0,
            "idempotency_key": "thread-B:abc123",
        }
    }
    for tid in ("thread-B", "other-thread"):  # khóa tường minh thắng thread_id
        asyncio.run(commit_erp_order(state, {"configurable": {"thread_id": tid}}))

    assert _stock(erp_sandbox) == 11
    assert _count(erp_sandbox, "erp_orders") == 1

    # Khóa khác -> đơn mới hợp lệ, trừ kho tiếp.
    state["pending_order"] = {**state["pending_order"], "idempotency_key": "thread-B:def456", "quantity": 1}
    asyncio.run(commit_erp_order(state, {"configurable": {"thread_id": "thread-B"}}))
    assert _stock(erp_sandbox) == 10
    assert _count(erp_sandbox, "erp_orders") == 2


def test_commit_erp_order_rejections_are_audited_and_do_not_deduct(erp_sandbox):
    from src.agent.nodes import commit_erp_order

    too_many = {"pending_order": {"sku": "AI-WS-X1", "quantity": 999}}
    unknown = {"pending_order": {"sku": "NOPE-1", "quantity": 1}}
    r1 = asyncio.run(commit_erp_order(too_many, {"configurable": {"thread_id": "t-x"}}))
    r2 = asyncio.run(commit_erp_order(unknown, {"configurable": {"thread_id": "t-y"}}))

    assert r1["approval_status"] == r2["approval_status"] == "rejected"
    assert _stock(erp_sandbox) == 14
    assert _count(erp_sandbox, "erp_orders") == 0
    with sqlite3.connect(erp_sandbox) as conn:
        actions = [r[0] for r in conn.execute("SELECT action FROM erp_audit_log ORDER BY log_id")]
    assert actions == ["REJECTED_INSUFFICIENT_STOCK", "REJECTED_SKU_NOT_FOUND"]


def test_commit_erp_order_migrates_legacy_table(erp_sandbox):
    """Bảng erp_orders cũ (chưa có idempotency_key) được migrate mà không mất dữ liệu."""
    from src.agent.nodes import commit_erp_order

    with sqlite3.connect(erp_sandbox) as conn:
        conn.execute(
            "CREATE TABLE erp_orders (order_id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT NOT NULL, "
            "quantity INTEGER NOT NULL, discount_percent REAL NOT NULL, total_vnd REAL NOT NULL, "
            "status TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO erp_orders (sku, quantity, discount_percent, total_vnd, status) VALUES ('AI-WS-X1', 1, 0, 1, 'COMMITTED')")
        conn.commit()

    state = {"pending_order": {"sku": "AI-WS-X1", "quantity": 1}}
    cfg = {"configurable": {"thread_id": "legacy"}}
    asyncio.run(commit_erp_order(state, cfg))
    asyncio.run(commit_erp_order(state, cfg))
    assert _count(erp_sandbox, "erp_orders") == 2  # 1 cũ + 1 mới
    assert _stock(erp_sandbox) == 13


# ----------------------------------------------------------------------
# MCP tool list_erp_orders (sandbox DB)
# ----------------------------------------------------------------------
def test_list_erp_orders_mcp_tool(erp_sandbox, monkeypatch):
    import mcp_server.project_mcp_server as srv
    from src.agent.nodes import commit_erp_order

    monkeypatch.setattr(srv, "DB_PATH", erp_sandbox)
    list_orders = unwrap_mcp_tool(srv.list_erp_orders)

    assert "Chưa có đơn hàng" in list_orders()  # DB chưa có bảng erp_orders

    for i in range(3):
        state = {"pending_order": {"sku": "AI-WS-X1", "quantity": 1, "idempotency_key": f"k{i}"}}
        asyncio.run(commit_erp_order(state, {"configurable": {"thread_id": f"th{i}"}}))
    asyncio.run(  # lặp lại: không thêm dòng
        commit_erp_order(
            {"pending_order": {"sku": "AI-WS-X1", "quantity": 1, "idempotency_key": "k2"}},
            {"configurable": {"thread_id": "th2"}},
        )
    )

    out = list_orders(limit=2)
    assert "2 đơn hàng ERP gần nhất" in out
    assert "key=k2" in out and "key=k1" in out and "key=k0" not in out
    assert out.index("key=k2") < out.index("key=k1")  # mới nhất trước
    assert "3 đơn hàng ERP gần nhất" in list_orders()
    assert _stock(erp_sandbox) == 11
