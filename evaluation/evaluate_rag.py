"""Script đánh giá chất lượng hệ thống RAG bằng RAGAS (chạy an toàn với Ollama Local)."""

import asyncio
from typing import Dict, List

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.agent.graph import agent_app
from src.agent.state import GraphState
from src.config import settings
from src.rag.indexing import build_or_load_vectorstore


TEST_SAMPLES: List[Dict[str, str]] = [
    {
        "question": "Hệ thống sử dụng mô hình Reranker nào và chạy trên phần cứng nào?",
        "ground_truth": (
            "Hệ thống sử dụng mô hình Cross-Encoder FlashRank ms-marco-MultiBERT-L-12 "
            "chạy trên CPU và giữ lại top_n=3 đoạn văn bản."
        ),
    },
    {
        "question": "Laptop AI Workstation X1 có giá niêm yết và thời gian bảo hành bao lâu?",
        "ground_truth": (
            "Laptop AI Workstation X1 (SKU: AI-WS-X1) có giá niêm yết 58,900,000 VNĐ "
            "và bảo hành 36 tháng chính hãng."
        ),
    },
]


async def _collect_rag_outputs():
    """Chạy tập câu hỏi kiểm thử qua LangGraph Agent để thu thập ngữ cảnh và câu trả lời."""
    build_or_load_vectorstore()
    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in TEST_SAMPLES:
        state: GraphState = {
            "question": sample["question"],
            "rewritten_query": sample["question"],
            "datasource": "vectorstore",
            "metadata_filter": None,
            "documents": [],
            "citations": [],
            "mcp_result": "",
            "generation": "",
            "hallucination_grade": "yes",
            "retry_count": 0,
        }
        result = await agent_app.ainvoke(state)
        questions.append(sample["question"])
        answers.append(result.get("generation", ""))
        contexts.append([doc.page_content for doc in result.get("documents", [])])
        ground_truths.append(sample["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main() -> None:
    """Chạy chấm điểm RAGAS với RunConfig(max_workers=1, timeout=180) chống nghẽn Ollama."""
    dataset = asyncio.run(_collect_rag_outputs())

    eval_llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
    eval_embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    run_config = RunConfig(max_workers=1, timeout=180)

    score = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )

    report_path = settings.data_dir.parent / "evaluation" / "ragas_evaluation_report.csv"
    df = score.to_pandas()
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n=== KẾT QUẢ CHẤM ĐIỂM RAGAS ===")
    print(score)
    print(f"\nĐã xuất báo cáo chi tiết ra: {report_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
