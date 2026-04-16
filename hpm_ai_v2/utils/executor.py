"""
PythonExecutor — sandboxed Python code execution.

Ported from experiment_unified_perception_action.py.
"""
from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple


def _eval_path_worker(code_str: str, inputs: List[Any], expected: List[Any], tolerance: float = 0.05) -> bool:
    """
    Module-level worker for ProcessPoolExecutor — must be picklable.
    Re-executes code_str in a fresh namespace and checks against expected outputs.
    """
    import numpy as np
    executor = PythonExecutor()
    results, _ = executor.run_batch(code_str, inputs)
    if len(results) != len(expected):
        return False
        
    def is_equal(r: Any, e: Any) -> bool:
        if isinstance(r, (int, float, np.float64, np.int64)) and isinstance(e, (int, float, np.float64, np.int64)):
            abs_diff = abs(float(r) - float(e))
            # Hybrid tolerance: abs_diff < tolerance * (1 + abs(e))
            return abs_diff < tolerance * (1.0 + abs(float(e)))
            
        if isinstance(r, np.ndarray) and isinstance(e, np.ndarray):
            return np.allclose(r, e, atol=tolerance, rtol=tolerance)
        if isinstance(r, list) and isinstance(e, list):
            if len(r) != len(e): return False
            return all(is_equal(ri, ei) for ri, ei in zip(r, e))
        try:
            return bool(r == e)
        except ValueError:
            if isinstance(r, (list, tuple, np.ndarray)) and isinstance(e, (list, tuple, np.ndarray)):
                return np.array_equal(r, e)
            return False

    for r, e in zip(results, expected):
        if not is_equal(r, e):
            return False
    return True


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
        # Use 'inputs' so the code can access the full batch if needed (e.g. for compositions)
        # But traditionally code strings use 'inp' for the current element.
        # Let's support both.
        code = (
            "def test_func(inp, inputs):\n"
            "    x = 0\n"
            "    val = 0\n"
            "    res = None\n"
            "    " + indented + "\n"
            "    return res\n"
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
                # Pass both the current element 'inp' and the full batch 'inputs'
                results.append(test_func(copy.deepcopy(inp), copy.deepcopy(inputs)))
                errors.append(None)
            except Exception as e:
                results.append(None)
                errors.append(type(e).__name__)
        return results, errors
