"""
physics.py - Minimal physics substrate for HPM agents.
Only provides dynamic library call and constant lookup.
No AST scanning, no hardcoded formula extraction.
"""

import importlib
import sympy
import scipy.constants as const
from typing import Dict, Any, List, Union

from .registry import ToolRegistry


def physics_library_call(function: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute any function from fphysics, sympy, or scipy.constants.
    Agent learns usage through trial and reward.
    """
    try:
        parts = function.split('.')
        module_name = '.'.join(parts[:-1])
        func_name = parts[-1]

        # Try to import from common physics libraries
        module = None
        for lib in ['fphysics', 'sympy', 'scipy.constants']:
            try:
                if module_name == lib:
                    module = importlib.import_module(lib)
                elif module_name.startswith(lib):
                    module = importlib.import_module(module_name)
                else:
                    module = importlib.import_module(f"{lib}.{module_name}")
                break
            except ImportError:
                continue
        
        if module is None:
            module = importlib.import_module(module_name)
        
        func = getattr(module, func_name)

        # Basic argument conversion
        converted_args = {}
        for k, v in args.items():
            if k == 'expr' and isinstance(v, str):
                converted_args[k] = sympy.sympify(v)
            elif isinstance(v, list):
                converted_args[k] = v  # Keep as list; function should handle
            else:
                converted_args[k] = v

        result = func(**converted_args)

        # Serialize
        if isinstance(result, sympy.Basic):
            result = str(result)
        elif hasattr(result, 'tolist'):
            result = result.tolist()
        elif not isinstance(result, (int, float, str, bool, list, dict)):
            result = str(result)

        return {"result": result, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


def list_physics_functions() -> Dict[str, Any]:
    """Dynamically discover functions in fphysics, sympy, and scipy.constants."""
    functions = []
    libs = ['fphysics', 'sympy', 'scipy.constants']
    import inspect
    
    for lib_name in libs:
        try:
            lib = importlib.import_module(lib_name)
            for name, obj in inspect.getmembers(lib):
                if not name.startswith('_') and inspect.isroutine(obj):
                    functions.append(f"{lib_name}.{name}")
        except ImportError:
            continue
    
    return {"functions": functions, "count": len(functions), "status": "success"}


def physics_constant_lookup(query: str) -> Dict[str, Any]:
    """Look up a physical constant from scipy.constants."""
    try:
        val = getattr(const, query)
        return {"value": val, "name": query, "status": "success"}
    except AttributeError:
        matches = const.find(query)
        if matches:
            matches.sort(key=len)
            best = matches[0]
            return {"value": const.value(best), "name": best, "status": "success"}
    return {"status": "not_found", "error": f"Constant '{query}' not found."}


def sympy_solve(equations: List[str], variables: List[str]) -> Dict[str, Any]:
    """Solve symbolic equations using SymPy."""
    try:
        parsed = []
        for eq in equations:
            if '=' in eq:
                lhs, rhs = eq.split('=')
                parsed.append(sympy.parse_expr(lhs) - sympy.parse_expr(rhs))
            else:
                parsed.append(sympy.parse_expr(eq))
        sols = sympy.solve(parsed, [sympy.Symbol(v) for v in variables], dict=True)
        return {"solutions": [{str(k): str(v) for k, v in s.items()} for s in sols], "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def numeric_eval(expr: str, substitutions: Dict[str, float]) -> Dict[str, Any]:
    """Evaluate a symbolic expression numerically."""
    try:
        e = sympy.parse_expr(expr)
        clean = {k: float(v) for k, v in substitutions.items()}
        return {"value": float(e.subs(clean)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def register_physics_tools():
    ToolRegistry.register(
        name="list_physics_functions",
        tool_fn=list_physics_functions,
        input_keys=[],
        output_key="catalog",
        cost=0.01,
        description="Dynamically discover physics-related functions."
    )
    ToolRegistry.register(
        name="physics_library_call",
        tool_fn=physics_library_call,
        input_keys=["function", "args"],
        output_key="result",
        cost=0.05,
        description="Execute a physics function from external libraries."
    )
    ToolRegistry.register(
        name="physics_constant_lookup",
        tool_fn=physics_constant_lookup,
        input_keys=["query"],
        output_key="constant",
        cost=0.01,
        description="Look up a physical constant from scipy.constants."
    )
    ToolRegistry.register(
        name="sympy_solve",
        tool_fn=sympy_solve,
        input_keys=["equations", "variables"],
        output_key="solutions",
        cost=0.05,
        description="Solve symbolic equations using SymPy."
    )
    ToolRegistry.register(
        name="numeric_eval",
        tool_fn=numeric_eval,
        input_keys=["expr", "substitutions"],
        output_key="value",
        cost=0.01,
        description="Evaluate a symbolic expression numerically."
    )
    print("[PhysicsTools] Registered minimal physics substrate tools.")


register_physics_tools()
