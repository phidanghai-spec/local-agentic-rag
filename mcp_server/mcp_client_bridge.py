"""Cầu nối tích hợp MCP Server tools vào LangGraph bằng langchain-mcp-adapters."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from src.config import settings


MCP_SERVER_PATH = Path(__file__).resolve().parent / "project_mcp_server.py"


def get_mcp_server_config() -> Dict[str, Any]:
    """Cấu hình kết nối stdio tới FastMCP server nội bộ."""
    return {
        "local_erp_tools": {
            "command": sys.executable,
            "args": [str(MCP_SERVER_PATH)],
            "transport": "stdio",
        }
    }


async def load_mcp_tools() -> List[BaseTool]:
    """Tải danh sách LangChain Tools từ FastMCP server."""
    client = MultiServerMCPClient(get_mcp_server_config())
    return await client.get_tools()


def _extract_fallback_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Trích xuất tool calls nếu model nhỏ xuất JSON trực tiếp trong content."""
    calls: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                calls.append({"name": obj["name"], "args": obj["arguments"]})
        except json.JSONDecodeError:
            continue

    if not calls:
        for match in re.findall(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}", content):
            try:
                obj = json.loads(match)
                if "name" in obj and "arguments" in obj:
                    calls.append({"name": obj["name"], "args": obj["arguments"]})
            except json.JSONDecodeError:
                continue
    return calls


def _normalize_tool_args(args: Any) -> Dict[str, Any]:
    """Làm sạch tham số nếu LLM lồng dict (vd: {'sku': {'sku': 'AI-WS-X1'}})."""
    if not isinstance(args, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, dict) and k in v and len(v) == 1:
            cleaned[k] = v[k]
        else:
            cleaned[k] = v
    return cleaned


async def execute_mcp_tool_query(question: str) -> str:
    """Chạy vòng lặp ReAct tối đa 4 bước để LLM gọi MCP tools tuần tự (tra cứu -> tính toán)."""
    tools = await load_mcp_tools()
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    ).bind_tools(tools)

    tool_map = {tool.name: tool for tool in tools}
    messages: List[Any] = [
        SystemMessage(
            content=(
                "Bạn là trợ lý kỹ thuật có quyền gọi công cụ MCP.\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. Nếu cần tra cứu giá hoặc tồn kho, hãy gọi `query_erp_inventory` TRƯỚC để lấy giá thật.\n"
                "2. Không được tự bịa `unit_price` khi chưa có kết quả từ `query_erp_inventory`.\n"
                "3. Sau khi có đơn giá thật, mới gọi `calculate_order_total` nếu cần tính tổng tiền."
            )
        ),
        HumanMessage(content=question),
    ]

    executed_outputs: List[str] = []

    for _ in range(4):
        ai_msg = await llm.ainvoke(messages)
        messages.append(ai_msg)

        raw_calls = list(getattr(ai_msg, "tool_calls", None) or [])
        if not raw_calls and isinstance(ai_msg.content, str):
            raw_calls = _extract_fallback_tool_calls(ai_msg.content)

        if not raw_calls:
            break

        for idx, call in enumerate(raw_calls):
            tool_name = call.get("name", "")
            tool_args = _normalize_tool_args(call.get("args", {}))
            tool_obj = tool_map.get(tool_name)

            if tool_obj is None:
                result_str = f"[Tool '{tool_name}' không tồn tại]"
            else:
                try:
                    result = await tool_obj.ainvoke(tool_args)
                    result_str = str(result)
                except Exception as exc:
                    result_str = f"[Lỗi thực thi {tool_name}: {exc}]"

            executed_outputs.append(f"[{tool_name}({tool_args})]: {result_str}")
            call_id = call.get("id") or f"call_{len(executed_outputs)}_{idx}"
            messages.append(ToolMessage(content=result_str, tool_call_id=call_id))

    return "\n".join(executed_outputs) if executed_outputs else str(messages[-1].content)
