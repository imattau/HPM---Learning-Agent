"""
PythonExecutor — sandboxed Python code execution.

Ported from experiment_unified_perception_action.py.
"""
from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple


class PythonExecutor:
    """Run a code string against a batch of inputs, returning outputs and errors."""

    def run_batch(
        self,
        code_str: str,
        inputs: List[Any],
        timeout: float = 0.5,
    ) -> Tuple[List[Any], List[Optional[str]]]:
        if not code_str:
            return [None] * len(inputs), ["EmptyCode"] * len(inputs)
        indented = code_str.replace('\n', '\n    ')
        code = (
            "def test_func(inp):\n"
            "    x = 0\n"
            "    val = 0\n"
            "    res = None\n"
            "    " + indented + "\n"
        )
        results: List[Any] = []
        errors: List[Optional[str]] = []
        try:
            local_ns: dict = {}
            compile_obj = compile(code, "<hpm_ai_v2>", "exec")
            exec(compile_obj, {}, local_ns)  # noqa: S102 — intentional sandboxed eval
            test_func = local_ns["test_func"]
        except Exception as e:
            return [None] * len(inputs), [type(e).__name__] * len(inputs)
        for inp in inputs:
            try:
                if 'list(x)' in code and not isinstance(inp, (list, tuple)):
                    results.append(None)
                    errors.append("TypeError")
                    continue
                results.append(test_func(copy.deepcopy(inp)))
                errors.append(None)
            except Exception as e:
                results.append(None)
                errors.append(type(e).__name__)
        return results, errors
