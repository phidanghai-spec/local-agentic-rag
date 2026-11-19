"""Định nghĩa StateGraph LangGraph kết nối Logical Routing, CRAG, Self-RAG & HITL Checkpointing."""

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    call_mcp_tools,
    check_hallucination,
    commit_erp_order,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
    route_after_mcp,
    route_question,
)
from src.agent.state import GraphState
from src.config import CHECKPOINT_DB_PATH, settings


def route_decision(state: GraphState) -> str:
    """Điều hướng từ Logical Router sang nhánh xử lý tương ứng."""
    datasource = state.get("datasource", "vectorstore")
    if datasource == "mcp_tools":
        return "call_mcp_tools"
    if datasource == "direct_answer":
        return "generate"
    return "retrieve"


def decide_after_grading(state: GraphState) -> str:
    """Quyết định sinh câu trả lời hay viết lại truy vấn (chặn retry_count <= 2)."""
    filtered_docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if filtered_docs:
        return "generate"

    if retry_count < settings.max_retries:
        return "rewrite_query"

    return "generate"


def build_workflow() -> StateGraph:
    """Định nghĩa cấu trúc 8 node của đồ thị (chưa biên dịch với Checkpointer)."""
    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("call_mcp_tools", call_mcp_tools)
    workflow.add_node("commit_erp_order", commit_erp_order)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)

    workflow.add_edge(START, "route_question")

    workflow.add_conditional_edges(
        "route_question",
        route_decision,
        {
            "retrieve": "retrieve",
            "call_mcp_tools": "call_mcp_tools",
            "generate": "generate",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_conditional_edges(
        "call_mcp_tools",
        route_after_mcp,
        {
            "commit_erp_order": "commit_erp_order",
            "generate": "generate",
        },
    )
    workflow.add_edge("commit_erp_order", "generate")
    workflow.add_edge("generate", "check_hallucination")
    workflow.add_edge("check_hallucination", END)

    return workflow


def compile_graph(checkpointer: Optional[Any] = None, enable_hitl: bool = True):
    """Biên dịch đồ thị với checkpointer bất kỳ (AsyncSqliteSaver hoặc None)."""
    workflow = build_workflow()
    interrupt_nodes = ["commit_erp_order"] if (enable_hitl and checkpointer is not None) else []
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )


# Đồ thị mặc định dùng cho CLI / đánh giá không cần Checkpointer bền vững
agent_app = compile_graph(checkpointer=None, enable_hitl=False)
