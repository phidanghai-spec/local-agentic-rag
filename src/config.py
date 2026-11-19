"""Cấu hình trung tâm cho hệ thống Local Agentic RAG."""

from pathlib import Path
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DB_PATH = BASE_DIR / "data" / "langgraph_checkpoints.sqlite"


class Settings(BaseModel):
    """Cấu hình mô hình, lưu trữ và siêu tham số truy xuất."""

    # Đường dẫn thư mục
    data_dir: Path = Field(default=BASE_DIR / "data")
    persist_directory: Path = Field(default=BASE_DIR / "chroma_db")
    sqlite_db_path: Path = Field(default=BASE_DIR / "data" / "erp_inventory.db")

    # Cấu hình Ollama Local
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="qwen2.5-coder:3b")
    embedding_model: str = Field(default="nomic-embed-text")
    llm_temperature: float = Field(default=0.0)

    # Cấu hình Vector Store (ChromaDB)
    collection_name: str = Field(default="local_agentic_rag")
    chroma_distance_space: str = Field(default="cosine")

    # Cấu hình Chunking
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)

    # Cấu hình Hybrid Search & Reranking
    bm25_k: int = Field(default=5)
    vector_k: int = Field(default=5)
    ensemble_weights: list[float] = Field(default=[0.4, 0.6])
    flashrank_model: str = Field(default="ms-marco-MultiBERT-L-12")
    flashrank_top_n: int = Field(default=6)

    # Cấu hình LangGraph Self-Correction
    max_retries: int = Field(default=2)


settings = Settings()
