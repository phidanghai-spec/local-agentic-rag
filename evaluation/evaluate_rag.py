"""Script đánh giá chất lượng hệ thống RAG bằng RAGAS (2 pha + cache + phân nhóm)."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
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


EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "golden_dataset.json"
CACHE_PATH = EVAL_DIR / "eval_runs_cache.json"
REPORT_CSV_PATH = EVAL_DIR / "ragas_evaluation_report.csv"
SUMMARY_CSV_PATH = EVAL_DIR / "ragas_category_summary.csv"

REFUSAL_MARKERS = (
    "không có thông tin",
    "tài liệu hiện tại không có",
    "không liên quan đến",
    "không liên quan tới",
    "không được đề cập",
    "không tìm thấy",
    "không đề cập",
    "không có trong tài liệu",
    "chưa có dữ liệu",
    "không nằm trong",
    "không hỗ trợ",
    "không sử dụng",
    "không có",
    "100% cục bộ",
    "nội bộ",
    "on-premise",
)


def load_dataset(category: str | None = None, limit: int | None = None) -> List[Dict[str, Any]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    if category:
        items = [x for x in items if x["category"] == category]
    if limit:
        items = items[:limit]
    return items


def load_cached_runs() -> Dict[str, Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_runs(cache: Dict[str, Dict[str, Any]]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


async def run_agent_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chạy đồ thị LangGraph thật với thread_id riêng biệt cho từng câu hỏi
    để tránh rò rỉ checkpoint cũ.
    """
    case_id = case["id"]
    unique_thread = f"ragas_eval_{case_id}_{int(time.time())}"
    config = {"configurable": {"thread_id": unique_thread}}

    initial_state: GraphState = {
        "question": case["question"],
        "rewritten_query": case["question"],
        "datasource": "vectorstore",
        "metadata_filter": None,
        "documents": [],
        "citations": [],
        "mcp_result": "",
        "generation": "",
        "hallucination_grade": "yes",
        "retry_count": 0,
    }

    start = time.perf_counter()
    final_state = await agent_app.ainvoke(initial_state, config=config)
    elapsed_s = round(time.perf_counter() - start, 2)

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    answer = str(_get(final_state, "generation", "") or "").strip()
    docs = _get(final_state, "documents", []) or []
    contexts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in docs
    ]

    return {
        "id": case_id,
        "category": case["category"],
        "target_domain": case["target_domain"],
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "answer": answer,
        "contexts": contexts,
        "num_contexts": len(contexts),
        "latency_s": elapsed_s,
    }


def evaluate_out_of_scope(
    oos_runs: List[Dict[str, Any]],
    case_lookup: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Chấm điểm khả năng từ chối an toàn & không bịa số liệu cho nhóm Ngoài phạm vi."""
    rows = []
    for run in oos_runs:
        ans_lower = run["answer"].lower()
        case_meta = case_lookup[run["id"]]
        forbidden = case_meta.get("forbidden_hallucination_terms", [])

        has_refusal = any(marker in ans_lower for marker in REFUSAL_MARKERS)
        hallucinated_terms = [t for t in forbidden if t.lower() in ans_lower]
        passed = has_refusal and len(hallucinated_terms) == 0

        rows.append(
            {
                "id": run["id"],
                "category": "out_of_scope",
                "question": run["question"],
                "answer": run["answer"],
                "num_contexts": run["num_contexts"],
                "latency_s": run["latency_s"],
                "refusal_passed": 1.0 if passed else 0.0,
                "hallucinated_terms": ", ".join(hallucinated_terms),
            }
        )
    return pd.DataFrame(rows)


def build_ragas_evaluators():
    """Khởi tạo LLM & Embeddings cục bộ qua Ollama với temperature=0 để chấm ổn định."""
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_ctx=3072,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    )
    return judge_llm, judge_embeddings


async def _collect_phase_1(
    cases: List[Dict[str, Any]],
    resume: bool,
) -> List[Dict[str, Any]]:
    build_or_load_vectorstore()
    cache = load_cached_runs() if resume else {}
    collected_runs: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        if resume and cid in cache:
            print(f"[{idx}/{len(cases)}] Tái sử dụng cache Pha 1 cho {cid} ({case['category']})")
            collected_runs.append(cache[cid])
            continue

        print(f"[{idx}/{len(cases)}] Đang chạy Agent cho {cid} ({case['category']}): {case['question']}")
        run_out = await run_agent_for_case(case)
        cache[cid] = run_out
        save_cached_runs(cache)
        collected_runs.append(run_out)
        print(f"    -> Xong trong {run_out['latency_s']}s | Chunks: {run_out['num_contexts']}")

    return collected_runs


def main():
    parser = argparse.ArgumentParser(description="Đánh giá RAGAS 10 câu đa dạng cho local-agentic-rag")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Lọc theo nhóm: single_aspect | multi_aspect | comparison | out_of_scope",
    )
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu để test nhanh")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tái sử dụng kết quả Pha 1 đã lưu trong eval_runs_cache.json",
    )
    args = parser.parse_args()

    cases = load_dataset(category=args.category, limit=args.limit)
    case_lookup = {c["id"]: c for c in cases}

    # PHA 1: CHẠY LANGGRAPH AGENT VÀ LƯU CACHE SAU MỖI CÂU
    collected_runs = asyncio.run(_collect_phase_1(cases, resume=args.resume))

    in_domain_runs = [r for r in collected_runs if r["category"] != "out_of_scope"]
    oos_runs = [r for r in collected_runs if r["category"] == "out_of_scope"]

    # PHA 2: CHẤM ĐIỂM RAGAS CHO NHÓM TRONG PHẠM VI (IN-DOMAIN)
    if in_domain_runs:
        print(f"\nĐang chấm RAGAS cho {len(in_domain_runs)} câu hỏi trong phạm vi (max_workers=1)...")
        hf_dataset = Dataset.from_dict(
            {
                "question": [r["question"] for r in in_domain_runs],
                "answer": [r["answer"] for r in in_domain_runs],
                "contexts": [r["contexts"] for r in in_domain_runs],
                "ground_truth": [r["ground_truth"] for r in in_domain_runs],
            }
        )

        judge_llm, judge_embeddings = build_ragas_evaluators()
        run_cfg = RunConfig(max_workers=1, timeout=240, max_retries=2)

        result = evaluate(
            dataset=hf_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_cfg,
        )

        ragas_df = result.to_pandas()
        ragas_df.insert(0, "id", [r["id"] for r in in_domain_runs])
        ragas_df.insert(1, "category", [r["category"] for r in in_domain_runs])
        ragas_df["num_contexts"] = [r["num_contexts"] for r in in_domain_runs]
        ragas_df["latency_s"] = [r["latency_s"] for r in in_domain_runs]

        ragas_df.to_csv(REPORT_CSV_PATH, index=False, encoding="utf-8-sig")

        metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        summary_df = ragas_df.groupby("category")[metric_cols].mean().round(4)
        overall_row = ragas_df[metric_cols].mean().round(4).to_frame().T
        overall_row.index = ["OVERALL_IN_DOMAIN"]
        summary_df = pd.concat([summary_df, overall_row])
        summary_df.to_csv(SUMMARY_CSV_PATH, encoding="utf-8-sig")

        print("\n=== BẢNG ĐIỂM RAGAS THEO NHÓM CÂU HỎI ===")
        print(summary_df.to_string())

    # PHA 3: CHẤM ĐIỂM CHỐNG ẢO GIÁC CHO NHÓM NGOÀI PHẠM VI (OUT-OF-SCOPE)
    if oos_runs:
        oos_df = evaluate_out_of_scope(oos_runs, case_lookup)
        oos_rate = oos_df["refusal_passed"].mean()
        print("\n=== KIỂM ĐỊNH CHỐNG ẢO GIÁC NGOÀI PHẠM VI (OUT-OF-SCOPE) ===")
        print(oos_df[["id", "refusal_passed", "hallucinated_terms", "latency_s"]].to_string(index=False))
        print(f"Tỷ lệ từ chối an toàn: {oos_rate:.2%}")

        oos_csv_path = EVAL_DIR / "out_of_scope_report.csv"
        oos_df.to_csv(oos_csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
