"""Bộ chấm điểm Quỹ đạo Hành động (Trajectory Eval) đối chiếu với Golden DAG."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCallStep(BaseModel):
    """Đại diện cho 1 bước gọi công cụ MCP thực tế."""

    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class GoldenTrajectory(BaseModel):
    """Định nghĩa Quỹ đạo Mẫu chuẩn (Golden DAG) cho 1 kịch bản kiểm thử."""

    case_id: str
    expected_nodes: List[str] = Field(
        description="Chuỗi node bắt buộc phải đi qua đúng thứ tự (Ordered Subsequence)"
    )
    forbidden_nodes: List[str] = Field(
        default_factory=list,
        description="Các node tuyệt đối không được kích hoạt ở kịch bản này",
    )
    expected_tools: List[str] = Field(
        default_factory=list,
        description="Chuỗi tên công cụ MCP phải gọi đúng thứ tự",
    )
    required_tool_args: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Tham số bắt buộc phải có khi gọi từng tool",
    )
    max_steps: int = Field(default=12, description="Ngưỡng chặn vòng lặp vô hạn")


class TrajectoryEvalReport(BaseModel):
    """Kết quả đối chiếu quỹ đạo thực tế với Golden DAG."""

    case_id: str
    passed: bool
    score: float
    node_order_matched: bool
    tool_order_matched: bool
    forbidden_violations: List[str] = Field(default_factory=list)
    arg_violations: List[str] = Field(default_factory=list)
    step_limit_exceeded: bool = False
    observed_nodes: List[str] = Field(default_factory=list)
    observed_tools: List[str] = Field(default_factory=list)


def _is_ordered_subsequence(expected: List[str], observed: List[str]) -> bool:
    """Kiểm tra `expected` xuất hiện đúng thứ tự trong `observed`."""
    if not expected:
        return True
    it = iter(observed)
    return all(item in it for item in expected)


def evaluate_trajectory(
    golden: GoldenTrajectory,
    observed_nodes: List[str],
    observed_tool_calls: Optional[List[ToolCallStep]] = None,
) -> TrajectoryEvalReport:
    """Chấm điểm quỹ đạo hành động của Agent đối chiếu với Golden DAG."""
    tool_calls = observed_tool_calls or []
    observed_tool_names = [t.name for t in tool_calls]

    node_order_matched = _is_ordered_subsequence(golden.expected_nodes, observed_nodes)

    tool_order_matched = (
        observed_tool_names == golden.expected_tools
        if golden.expected_tools
        else len(observed_tool_names) == 0
    )

    forbidden_violations = [n for n in observed_nodes if n in golden.forbidden_nodes]

    arg_violations: List[str] = []
    for tool_name, expected_kv in golden.required_tool_args.items():
        matching_calls = [t for t in tool_calls if t.name == tool_name]
        if not matching_calls:
            arg_violations.append(f"Thiếu lời gọi tool `{tool_name}`")
            continue
        call_args = matching_calls[0].args
        for k, expected_val in expected_kv.items():
            if call_args.get(k) != expected_val:
                arg_violations.append(
                    f"Tool `{tool_name}` sai tham số `{k}`: kỳ vọng `{expected_val}`, thực tế `{call_args.get(k)}`"
                )

    step_limit_exceeded = len(observed_nodes) > golden.max_steps

    checks = [
        node_order_matched,
        tool_order_matched,
        len(forbidden_violations) == 0,
        len(arg_violations) == 0,
        not step_limit_exceeded,
    ]
    score = round(sum(1 for ok in checks if ok) / len(checks), 4)
    passed = all(checks)

    return TrajectoryEvalReport(
        case_id=golden.case_id,
        passed=passed,
        score=score,
        node_order_matched=node_order_matched,
        tool_order_matched=tool_order_matched,
        forbidden_violations=forbidden_violations,
        arg_violations=arg_violations,
        step_limit_exceeded=step_limit_exceeded,
        observed_nodes=observed_nodes,
        observed_tools=observed_tool_names,
    )
