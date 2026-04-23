"""
TaskPerceptor — innate, always-on input classifier for HPM agents.
Runs before tool selection every episode. No external dependencies.
"""
import re
from typing import Any, Dict, List

# Priority order: expression > list > boolean > string > numeric > mixed
_OPERATOR_RE = re.compile(r'\d\s*[+\-*/]\s*\d')
_LIST_RE = re.compile(r'\[.*?\]|(\d+\s*,\s*){2,}\d+')
_BOOL_STARTERS = re.compile(r'^(is|does|are|can|has|have|will|was|were|did)\b', re.IGNORECASE)
_STRING_KEYWORDS = re.compile(
    r'\b(palindrome|string|character|word|letter|sentence|upper|lower|reverse|split|join|strip|capitalize|replace|substr|prefix|suffix|concat)\b',
    re.IGNORECASE
)
_NUMBER_RE = re.compile(r'-?\d+\.?\d*')

_OPERATION_PATTERNS = [
    ("compute",   re.compile(r'\b(calculate|compute|what\s+is)\b', re.IGNORECASE)),
    ("classify",  re.compile(r'\b(determine|identify|type\s+of)\b', re.IGNORECASE)),
    ("extract",   re.compile(r'\b(find|get|count|list|extract)\b', re.IGNORECASE)),
    ("compare",   re.compile(r'\b(compare|greater|less|equal|difference|larger|smaller)\b', re.IGNORECASE)),
    ("transform", re.compile(r'\b(split|join|reverse|sort|upper|lower|strip|replace)\b', re.IGNORECASE)),
]


class TaskPerceptor:
    """Classify task text into a structured percept dict."""

    def perceive(self, text: Any) -> Dict:
        """Return percept dict for any input. Never raises."""
        s = str(text).strip() if text is not None else ""
        tokens = s.lower().split() if s else []
        numeric_values = [float(n) for n in _NUMBER_RE.findall(s)]
        is_question = s.endswith("?")

        input_type = self._detect_type(s, numeric_values)
        operation = self._detect_operation(s)

        return {
            "input_type": input_type,
            "operation": operation,
            "numeric_values": numeric_values,
            "tokens": tokens,
            "is_question": is_question,
        }

    def _detect_type(self, s: str, numeric_values: List[float]) -> str:
        if not s:
            return "mixed"
        if _OPERATOR_RE.search(s):
            return "expression"
        if _LIST_RE.search(s):
            return "list"
        if _BOOL_STARTERS.match(s) and not _STRING_KEYWORDS.search(s):
            return "boolean"
        if _STRING_KEYWORDS.search(s):
            return "string"
        stripped_nums = _NUMBER_RE.sub("", s).strip()
        non_numeric_words = [w for w in stripped_nums.split() if w not in (",", ".", "and", "the")]
        if numeric_values and len(non_numeric_words) == 0:
            return "numeric"
        if numeric_values:
            return "mixed"
        return "mixed"

    def _detect_operation(self, s: str) -> str:
        for op_name, pattern in _OPERATION_PATTERNS:
            if pattern.search(s):
                return op_name
        return "compute"
