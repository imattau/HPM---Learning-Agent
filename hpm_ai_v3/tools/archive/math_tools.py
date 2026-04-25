"""
math_tools.py - Minimal math substrate for HPM agents.
Only provides evaluate_at, math_library_call, and dynamic function listing.
"""

import importlib
import inspect
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Tuple, Callable, Optional

from .registry import ToolRegistry


class EnvironmentQuery:
    """Wrapper for the hidden environment. The agent doesn't see the true function."""
    def __init__(self, true_fn: Callable, x_range: Tuple[float, float] = (-5, 5)):
        self.true_fn = true_fn
        self.x_range = x_range
        self.query_count = 0

    def evaluate(self, x: float) -> float:
        self.query_count += 1
        return self.true_fn(x)


_ENV: Optional[EnvironmentQuery] = None


def set_environment(true_fn: Callable, x_range: Tuple[float, float] = (-5, 5)):
    global _ENV
    _ENV = EnvironmentQuery(true_fn, x_range)


def get_query_count() -> int:
    return _ENV.query_count if _ENV else 0


def evaluate_at(x: float) -> Dict[str, Any]:
    """Query the hidden function at a specific point."""
    if _ENV is None:
        return {"error": "Environment not set", "status": "failed"}
    y = _ENV.evaluate(x)
    return {"x": x, "y": y, "status": "success"}


def list_math_functions() -> Dict[str, Any]:
    """
    Dynamically discover all callable functions in numpy and sympy.
    Returns only function names—no descriptions, no categories.
    """
    functions = []
    modules = [('numpy', np), ('sympy', sp)]
    
    for module_name, module in modules:
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
            if inspect.isroutine(obj):  # functions and built-ins
                functions.append(f"{module_name}.{name}")
            elif inspect.ismodule(obj) and obj.__name__.startswith(module_name):
                # Optionally recurse into submodules (simplified: just top-level for now)
                pass
    
    return {"functions": functions, "count": len(functions), "status": "success"}


def math_library_call(function: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute any function from numpy or sympy with given arguments.
    Agent learns usage through trial and reward.
    """
    try:
        parts = function.split('.')
        module_name = '.'.join(parts[:-1])
        func_name = parts[-1]

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        converted_args = {}
        for k, v in args.items():
            if k in ('x', 'y', 'points', 'data') and isinstance(v, list):
                if k == 'points':
                    xs = [p[0] for p in v]
                    ys = [p[1] for p in v]
                    converted_args['x'] = np.array(xs)
                    converted_args['y'] = np.array(ys)
                else:
                    converted_args[k] = np.array(v)
            elif k == 'expr' and isinstance(v, str):
                converted_args[k] = sp.sympify(v)
            elif k == 'deg' and isinstance(v, (int, float)):
                converted_args[k] = int(v)
            else:
                converted_args[k] = v

        result = func(**converted_args)

        # Serialize result
        if isinstance(result, np.ndarray):
            result = result.tolist()
        elif isinstance(result, sp.Basic):
            result = str(result)
        elif not isinstance(result, (int, float, str, bool, list, dict)):
            result = str(result)

        return {"result": result, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


def register_math_tools():
    ToolRegistry.register(
        name="evaluate_at",
        tool_fn=evaluate_at,
        input_keys=["x"],
        output_key="result",
        cost=0.1,
        description="Query the hidden function at a specific x value."
    )
    ToolRegistry.register(
        name="list_math_functions",
        tool_fn=list_math_functions,
        input_keys=[],
        output_key="catalog",
        cost=0.01,
        description="Get all available function names from numpy and sympy."
    )
    ToolRegistry.register(
        name="math_library_call",
        tool_fn=math_library_call,
        input_keys=["function", "args"],
        output_key="result",
        cost=0.05,
        description="Execute any function from numpy or sympy with given arguments."
    )
    print("[MathTools] Registered agnostic math substrate tools.")


register_math_tools()
