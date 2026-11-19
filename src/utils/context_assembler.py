"""Module Dynamic Context Assembler kiểm soát ngân sách token trước khi gọi LLM."""

import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class TokenBudgetConfig(BaseModel):
    """Cấu hình ngân sách Token tối ưu cho Qwen2.5-Coder 3B trên iGPU."""

    max_context_window: int = Field(default=3072, description="Tổng cửa sổ ngữ cảnh an toàn")
    generation_reserve: int = Field(default=512, description="Token dự trữ cho câu trả lời")
    system_prompt_budget: int = Field(default=420, description="Hạn mức cho System Prompt")
    tool_schema_budget: int = Field(default=350, description="Hạn mức cho MCP Tool Schemas")
    history_budget: int = Field(default=300, description="Hạn mức cho lịch sử hội thoại gần nhất")

    @property
    def available_rag_budget(self) -> int:
        used = (
            self.generation_reserve
            + self.system_prompt_budget
            + self.tool_schema_budget
            + self.history_budget
        )
        return max(256, self.max_context_window - used)


class DynamicContextAssembler:
    """Bộ lắp ráp ngữ cảnh động kiểm soát chặt chẽ ngân sách token."""

    def __init__(self, config: TokenBudgetConfig | None = None):
        self.config = config or TokenBudgetConfig()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Ước lượng token chính xác cho tiếng Việt + Code Python/Markdown chạy 100% offline.
        Trung bình 1 từ tiếng Việt hoặc ký hiệu code tương đương ~1.35 token trên Qwen tokenizer.
        """
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return int(len(tokens) * 1.35) + 4

    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Cắt tỉa văn bản an toàn nếu vượt quá hạn mức."""
        if self.estimate_tokens(text) <= max_tokens:
            return text
        words = text.split()
        safe_word_limit = max(10, int(max_tokens / 1.4))
        return " ".join(words[:safe_word_limit]) + " ... [Đã cắt tỉa theo ngân sách]"

    def assemble_rag_context(
        self,
        query: str,
        docs: List[Document],
        system_prompt: str = "",
        has_tools: bool = False,
    ) -> Tuple[str, List[Document], Dict[str, Any]]:
        """
        Đóng gói tài liệu truy xuất theo điểm số FlashRank và hạn mức token thực tế.
        Trả về: (formatted_context_str, kept_documents, telemetry_metrics)
        """
        sys_tokens = self.estimate_tokens(system_prompt)
        query_tokens = self.estimate_tokens(query)
        tool_tokens = self.config.tool_schema_budget if has_tools else 0

        dynamic_rag_budget = (
            self.config.max_context_window
            - self.config.generation_reserve
            - sys_tokens
            - query_tokens
            - tool_tokens
        )
        dynamic_rag_budget = max(256, dynamic_rag_budget)

        kept_docs: List[Document] = []
        formatted_blocks: List[str] = []
        used_rag_tokens = 0

        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "internal_doc")
            score = float(doc.metadata.get("relevance_score", 0.0))
            header = f"[Tài liệu #{idx} | Nguồn: {source} | Độ tin cậy: {score:.2f}]\n"
            block_text = header + doc.page_content.strip()
            block_tokens = self.estimate_tokens(block_text)

            if used_rag_tokens + block_tokens <= dynamic_rag_budget:
                kept_docs.append(doc)
                formatted_blocks.append(block_text)
                used_rag_tokens += block_tokens
            else:
                remaining = dynamic_rag_budget - used_rag_tokens
                if idx == 1 and remaining > 120:
                    truncated_content = self.truncate_to_budget(
                        doc.page_content.strip(), remaining - 30
                    )
                    kept_docs.append(doc)
                    formatted_blocks.append(header + truncated_content)
                    used_rag_tokens += self.estimate_tokens(header + truncated_content)
                break

        telemetry = {
            "total_input_docs": len(docs),
            "kept_docs_count": len(kept_docs),
            "rag_tokens_used": used_rag_tokens,
            "rag_tokens_budget": dynamic_rag_budget,
        }

        return "\n\n---\n\n".join(formatted_blocks), kept_docs, telemetry
