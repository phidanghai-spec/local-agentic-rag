"""Middleware ẩn danh PII (Việt Nam) & Chặn Prompt Injection liên tác nhân."""

import re
import uuid
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class PIIScanResult(BaseModel):
    """Kết quả chuẩn hóa từ lớp PII & Security Proxy."""

    original_text: str
    masked_text: str = Field(
        description="Văn bản đã thay PII bằng mã UUID an toàn để đưa vào LangGraph"
    )
    cache_safe_text: str = Field(
        description="Văn bản chuẩn hóa nhãn chung để kiểm tra/lưu cache an toàn"
    )
    vault: Dict[str, str] = Field(
        default_factory=dict,
        description="Bảng ánh xạ tạm thời trên RAM (Placeholder -> Giá trị gốc)",
    )
    detected_types: List[str] = Field(default_factory=list)
    has_pii: bool = False
    is_injection_blocked: bool = False
    injection_reason: str = ""


class PIIProxy:
    """
    Middleware ẩn danh PII (SĐT VN, Email, CCCD/CMND, Thẻ thanh toán)
    và chặn Prompt Injection trước khi đẩy dữ liệu vào Semantic Cache & LangGraph.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    )
    CARD_PATTERN = re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)")
    CCCD_12_PATTERN = re.compile(r"(?<!\d)0(?:0[1-9]|[1-8]\d|9[0-6])\d{9}(?!\d)")
    VN_PHONE_PATTERN = re.compile(
        r"(?<!\d)(?:\+84|84|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-9])(?:\d[\s.-]?\d{3}[\s.-]?\d{3}|[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)"
    )
    CMND_9_CONTEXT_PATTERN = re.compile(
        r"(?i)(?P<prefix>\b(?:cmnd|cccd|căn cước|chứng minh(?:\s+nhân\s+dân)?)\s*(?:số|:|-)?\s*)(?P<num>\d{9})\b"
    )

    INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
        (
            "override_instructions",
            re.compile(
                r"(?i)(ignore\s+(all\s+)?previous\s+instructions|bỏ\s+qua\s+(mọi\s+)?chỉ\s+thị\s+trước)"
            ),
        ),
        (
            "system_prompt_leak",
            re.compile(
                r"(?i)(reveal|print|show|in\s+ra|tiết\s+lộ).{0,25}(system\s+prompt|chỉ\s+thị\s+hệ\s+thống)"
            ),
        ),
        (
            "sql_mcp_injection",
            re.compile(
                r"(?i)(\bdrop\s+table\b|\bdelete\s+from\b|\bunion\s+select\b|;\s*--\s*$)"
            ),
        ),
    ]

    def _detect_injection(self, text: str) -> Tuple[bool, str]:
        for rule_name, pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                return True, f"Phát hiện dấu hiệu tấn công câu lệnh ({rule_name})"
        return False, ""

    def mask(self, text: str) -> PIIScanResult:
        """Quét Prompt Injection và tráo đổi toàn bộ PII sang mã UUID tạm thời."""
        is_blocked, reason = self._detect_injection(text)
        if is_blocked:
            return PIIScanResult(
                original_text=text,
                masked_text="",
                cache_safe_text="",
                is_injection_blocked=True,
                injection_reason=reason,
            )

        vault: Dict[str, str] = {}
        detected_types: List[str] = []
        masked = text
        cache_safe = text

        def _replace_entity(
            label: str, pattern: re.Pattern, current_masked: str, current_cache: str
        ) -> Tuple[str, str]:
            def _sub_fn(match: re.Match) -> str:
                raw_val = match.group(0)
                token_id = uuid.uuid4().hex[:6]
                placeholder = f"[PII_{label}_{token_id}]"
                vault[placeholder] = raw_val
                if label not in detected_types:
                    detected_types.append(label)
                return placeholder

            new_masked = pattern.sub(_sub_fn, current_masked)
            new_cache = pattern.sub(f"[PII_{label}]", current_cache)
            return new_masked, new_cache

        for label, pattern in (
            ("EMAIL", self.EMAIL_PATTERN),
            ("CARD", self.CARD_PATTERN),
            ("CCCD", self.CCCD_12_PATTERN),
            ("PHONE", self.VN_PHONE_PATTERN),
        ):
            masked, cache_safe = _replace_entity(label, pattern, masked, cache_safe)

        def _sub_cmnd(match: re.Match) -> str:
            prefix = match.group("prefix")
            raw_num = match.group("num")
            token_id = uuid.uuid4().hex[:6]
            placeholder = f"[PII_CMND_{token_id}]"
            vault[placeholder] = raw_num
            if "CMND" not in detected_types:
                detected_types.append("CMND")
            return f"{prefix}{placeholder}"

        masked = self.CMND_9_CONTEXT_PATTERN.sub(_sub_cmnd, masked)
        cache_safe = self.CMND_9_CONTEXT_PATTERN.sub(r"\g<prefix>[PII_CMND]", cache_safe)

        return PIIScanResult(
            original_text=text,
            masked_text=masked,
            cache_safe_text=cache_safe,
            vault=vault,
            detected_types=detected_types,
            has_pii=len(vault) > 0,
            is_injection_blocked=False,
        )

    @staticmethod
    def unmask(text: str, vault: Dict[str, str]) -> str:
        """Khôi phục lại giá trị gốc cho phản hồi cuối cùng trả về đúng người dùng."""
        if not text or not vault:
            return text
        restored = text
        for placeholder, original_val in vault.items():
            restored = restored.replace(placeholder, original_val)
        return restored


pii_proxy = PIIProxy()
