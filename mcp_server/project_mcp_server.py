"""FastMCP Server cung cấp các công cụ thực tế (SQLite, Tính toán, Đọc tài liệu)."""

import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "erp_inventory.db"

mcp = FastMCP("LocalAgenticRAG-Tools")


def _init_sqlite_db() -> None:
    """Khởi tạo cơ sở dữ liệu SQLite mẫu nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stock_qty INTEGER NOT NULL,
                unit_price_vnd REAL NOT NULL,
                warehouse TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO inventory (sku, name, stock_qty, unit_price_vnd, warehouse)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("AI-WS-X1", "Laptop AI Workstation X1", 14, 58900000, "Kho Hà Nội"),
                ("SSD-ENT-4T", "Bộ nhớ Ngoài NVMe Enterprise 4TB", 42, 9500000, "Kho TP.HCM"),
                ("GPU-RTX-4090", "Card đồ họa RTX 4090 24GB", 5, 52000000, "Kho TP.HCM"),
            ],
        )
        conn.commit()


_init_sqlite_db()


@mcp.tool()
def query_erp_inventory(sku: str) -> str:
    """Tra cứu tồn kho, đơn giá và vị trí kho hàng của sản phẩm theo mã SKU trong SQLite."""
    clean_sku = sku.strip().upper()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT sku, name, stock_qty, unit_price_vnd, warehouse FROM inventory WHERE sku = ?",
            (clean_sku,),
        ).fetchone()

    if not row:
        return f"Không tìm thấy sản phẩm với mã SKU '{clean_sku}' trong kho SQLite."

    return (
        f"SKU: {row[0]} | Tên: {row[1]} | "
        f"Tồn kho: {row[2]} chiếc | Đơn giá: {row[3]:,.0f} VNĐ | Vị trí: {row[4]}"
    )


@mcp.tool()
def calculate_order_total(unit_price: float, quantity: int, discount_percent: float = 0.0) -> str:
    """Tính chính xác tổng tiền đơn hàng có áp dụng chiết khấu (tránh sai số khi LLM tự tính nhẩm)."""
    if quantity <= 0 or unit_price < 0:
        return "Lỗi tham số: Số lượng phải > 0 và đơn giá không âm."
    if not (0.0 <= discount_percent <= 100.0):
        return "Lỗi tham số: Phần trăm chiết khấu phải từ 0 đến 100."

    subtotal = unit_price * quantity
    discount_amount = subtotal * (discount_percent / 100.0)
    final_total = subtotal - discount_amount
    return (
        f"Tạm tính: {subtotal:,.0f} VNĐ | "
        f"Chiết khấu ({discount_percent}%): -{discount_amount:,.0f} VNĐ | "
        f"Thành tiền: {final_total:,.0f} VNĐ"
    )


@mcp.tool()
def list_erp_orders(limit: int = 20) -> str:
    """Liệt kê các đơn hàng ERP đã chốt gần nhất (kèm khóa idempotency) từ SQLite."""
    limit = max(1, min(int(limit), 200))
    try:
        with sqlite3.connect(DB_PATH) as conn:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(erp_orders)").fetchall()}
            if not columns:
                return "Chưa có đơn hàng ERP nào được chốt."
            idem_col = "idempotency_key" if "idempotency_key" in columns else "NULL"
            rows = conn.execute(
                f"SELECT order_id, sku, quantity, discount_percent, total_vnd, status, created_at, {idem_col} "
                "FROM erp_orders ORDER BY order_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except sqlite3.Error as exc:
        return f"Lỗi đọc bảng erp_orders: {exc}"

    if not rows:
        return "Chưa có đơn hàng ERP nào được chốt."

    lines = [f"{len(rows)} đơn hàng ERP gần nhất:"]
    for oid, sku, qty, disc, total, status, created, idem in rows:
        lines.append(
            f"#{oid} | {sku} x{qty} | CK {disc}% | {total:,.0f} VNĐ | {status} | "
            f"{created} | key={idem or '-'}"
        )
    return "\n".join(lines)


@mcp.tool()
def read_technical_doc(filename: str) -> str:
    """Đọc toàn bộ nội dung một file tài liệu .md trong thư mục data/ một cách an toàn."""
    safe_name = Path(filename).name
    target_path = (DATA_DIR / safe_name).resolve()

    if target_path.parent != DATA_DIR.resolve() or not target_path.exists():
        available = [p.name for p in DATA_DIR.glob("*.md")]
        return f"Không tìm thấy tệp '{safe_name}'. Các tệp hiện có: {available}"

    return target_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")
