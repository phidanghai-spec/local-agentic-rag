"""Mini-LangSmith offline: lưu vết (trace) mỗi lượt chat/duyệt đơn vào SQLite cục bộ."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import BASE_DIR

DEFAULT_TELEMETRY_DB_PATH = BASE_DIR / "data" / "telemetry.sqlite"

_QUESTION_PREVIEW_CHARS = 120


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Percentile nội suy tuyến tính (giống numpy mặc định) trên dãy đã sắp xếp."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


class TelemetryStore:
    """Kho telemetry SQLite (không cần mạng/LangSmith)."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_TELEMETRY_DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT PRIMARY KEY,
                        thread_id TEXT,
                        question_preview TEXT,
                        datasource TEXT,
                        cache_hit INTEGER NOT NULL DEFAULT 0,
                        pii_tags TEXT NOT NULL DEFAULT '[]',
                        hitl_status TEXT NOT NULL DEFAULT 'none',
                        total_latency_ms REAL NOT NULL DEFAULT 0,
                        node_latencies TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at)")
                conn.commit()
            finally:
                conn.close()

    def record_trace(
        self,
        thread_id: str = "",
        question: str = "",
        datasource: Optional[str] = None,
        cache_hit: bool = False,
        pii_tags: Optional[Sequence[str]] = None,
        hitl_status: str = "none",
        total_latency_ms: float = 0.0,
        node_latencies: Optional[Dict[str, float]] = None,
        trace_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Ghi một trace; trả về trace_id. Chỉ lưu bản xem trước (đã ẩn danh) của câu hỏi."""
        trace_id = trace_id or uuid.uuid4().hex
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        preview = (question or "")[:_QUESTION_PREVIEW_CHARS]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO traces (
                        trace_id, thread_id, question_preview, datasource, cache_hit,
                        pii_tags, hitl_status, total_latency_ms, node_latencies, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        thread_id,
                        preview,
                        datasource,
                        1 if cache_hit else 0,
                        json.dumps(list(pii_tags or []), ensure_ascii=False),
                        hitl_status or "none",
                        float(total_latency_ms),
                        json.dumps(
                            {k: round(float(v), 2) for k, v in (node_latencies or {}).items()},
                            ensure_ascii=False,
                        ),
                        created_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return trace_id

    def get_metrics(self) -> Dict[str, Any]:
        """Tổng hợp: total_requests, cache_hit_rate, p50/p95 latency, hitl_counts."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT cache_hit, hitl_status, total_latency_ms FROM traces"
                ).fetchall()
            finally:
                conn.close()

        total = len(rows)
        latencies = sorted(float(r["total_latency_ms"]) for r in rows)
        hitl_counts: Dict[str, int] = {}
        for r in rows:
            hitl_counts[r["hitl_status"]] = hitl_counts.get(r["hitl_status"], 0) + 1
        hits = sum(1 for r in rows if r["cache_hit"])
        return {
            "total_requests": total,
            "cache_hit_rate": round(hits / total, 4) if total else 0.0,
            "p50_latency_ms": round(_percentile(latencies, 50), 2),
            "p95_latency_ms": round(_percentile(latencies, 95), 2),
            "hitl_counts": hitl_counts,
        }

    def list_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Các trace mới nhất trước."""
        limit = max(1, min(int(limit), 500))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM traces ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()

        traces: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["cache_hit"] = bool(item["cache_hit"])
            item["pii_tags"] = json.loads(item["pii_tags"] or "[]")
            item["node_latencies"] = json.loads(item["node_latencies"] or "{}")
            traces.append(item)
        return traces
