"""Fixtures và Mock Runnables dùng chung cho bộ kiểm thử CI không cần Ollama."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from evaluation.trajectory_eval import ToolCallStep


class _FakeStructuredRunnable(Runnable):
    """Giả lập đầu ra Pydantic chuẩn cho llm.with_structured_output(Schema)."""

    def __init__(self, schema: Type[BaseModel], resolver: Callable[[Type[BaseModel], Any], Any]):
        self.schema = schema
        self.resolver = resolver

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        result = self.resolver(self.schema, input)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, self.schema):
            return result
        if isinstance(result, dict):
            return self.schema(**result)
        return result

    async def ainvoke(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        return self.invoke(input, config=config, **kwargs)


class FakeDeterministicLLM(Runnable):
    """
    Fake LLM tương thích hoàn toàn với LangChain LCEL (prompt | llm)
    và with_structured_output() (cả sync & async) để chạy đồ thị thật trong CI.
    """

    def __init__(
        self,
        text_resolver: Callable[[Any], str],
        structured_resolver: Optional[Callable[[Type[BaseModel], Any], Any]] = None,
    ):
        self._text_resolver = text_resolver
        self._structured_resolver = structured_resolver or (lambda schema, _: schema())

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self._text_resolver(input))

    async def ainvoke(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AIMessage:
        return self.invoke(input, config=config, **kwargs)

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> Runnable:
        return _FakeStructuredRunnable(schema=schema, resolver=self._structured_resolver)


def unwrap_mcp_tool(tool_obj: Any) -> Callable[..., Any]:
    """Bóc hàm Python thật từ đối tượng @mcp.tool() của FastMCP."""
    return getattr(tool_obj, "fn", tool_obj)


class RealMCPToolSpy:
    """
    Gọi trực tiếp hàm thật trong mcp_server/project_mcp_server.py
    đồng thời ghi lại danh sách ToolCallStep để Trajectory Eval đối chiếu Golden DAG.
    """

    def __init__(self):
        import mcp_server.project_mcp_server as srv

        self.registry: Dict[str, Callable[..., Any]] = {
            "query_erp_inventory": unwrap_mcp_tool(srv.query_erp_inventory),
            "calculate_order_total": unwrap_mcp_tool(srv.calculate_order_total),
            "read_technical_doc": unwrap_mcp_tool(srv.read_technical_doc),
        }
        self.recorded_calls: List[ToolCallStep] = []

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.registry:
            raise KeyError(f"Tool MCP không tồn tại trong server thật: {tool_name}")
        fn = self.registry[tool_name]
        sig_params = list(inspect.signature(fn).parameters.keys())
        bound_kwargs: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in sig_params:
                bound_kwargs[k] = v
            elif k == "sku" and sig_params:
                bound_kwargs[sig_params[0]] = v
            elif k == "discount_percent":
                alt = next((p for p in sig_params if "discount" in p), None)
                bound_kwargs[alt or k] = v
            elif k == "unit_price":
                alt = next((p for p in sig_params if "price" in p), None)
                bound_kwargs[alt or k] = v

        self.recorded_calls.append(ToolCallStep(name=tool_name, args=dict(kwargs)))
        return fn(**bound_kwargs)
