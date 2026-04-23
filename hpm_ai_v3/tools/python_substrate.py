"""
python_substrate.py - Unified substrate for dynamic Python module exploration and execution.
Minimal HPM implementation: thin wrapper over module/function calls.
"""

import importlib
import inspect
from typing import Dict, Any, List, Optional
from .registry import ToolRegistry


def list_modules(filter_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a curated list of top-level modules for exploration."""
    modules = [
        "math", "numpy", "sympy", "scipy.constants", "re", "json", "spacy",
        "operator", "builtins", "textblob"
    ]
    if filter_list:
        modules = [m for m in modules if m in filter_list]
    return {"result": modules, "status": "success"}


# GLOBAL BLACKLIST for safe autonomous exploration
PYTHON_BLACKLIST = {
    "breakpoint", "input", "help", "exit", "quit", "eval", "exec", "open", 
    "getattr", "setattr", "delattr", "compile", "init_session", "init_printing",
    "pager_print", "preview", "pprint", "display", "interact", "interactive",
    "download"
}

def list_functions(module: str) -> Dict[str, Any]:
    """Dynamically discover all callable functions in a module with their signatures."""
    try:
        mod = importlib.import_module(module)
        functions = []
        
        for name, func in inspect.getmembers(mod, inspect.isroutine):
            if name.startswith("_") or name in PYTHON_BLACKLIST: continue
            
            params = []
            try:
                sig = inspect.signature(func)
                for p_name, param in sig.parameters.items():
                    hint = "Any"
                    if param.annotation != inspect.Parameter.empty:
                        if hasattr(param.annotation, "__name__"):
                            hint = param.annotation.__name__
                        else:
                            hint = str(param.annotation)
                    params.append({"name": p_name, "type": hint})
            except:
                params = [] 
                
            functions.append({"name": name, "parameters": params})
            
        return {"result": sorted(functions, key=lambda x: x['name']), "module": module, "count": len(functions), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def python_call(module: str, function: str, args: Any = None) -> Dict[str, Any]:
    """
    Agnostic execution: import a module and call a function with arguments.
    Supports dict (keyword), list (positional), or single value.
    """
    if function in PYTHON_BLACKLIST:
        return {"error": f"Function '{function}' is blacklisted for safety.", "status": "failed"}

    import numpy as np
    import sympy as sp
    
    def try_float_recursive(v):
        if isinstance(v, str):
            # First try direct float conversion (Fix 3: accept bare numbers)
            try: return float(v)
            except: 
                # If not a bare number, try symbolic evaluation if operators are present
                if any(op in v for op in ["log", "sqrt", "*", "/", "+", "-", "**"]):
                    try: return float(sp.sympify(v))
                    except: pass
                return v
        if isinstance(v, list): return [try_float_recursive(x) for x in v]
        if isinstance(v, dict): return {k: try_float_recursive(x) for k, x in v.items()}
        return v

    try:
        # Pre-emptive type alignment for foundational math
        if module in ["math", "numpy", "operator"] or function in ["float", "int", "sympify"]:
            args = try_float_recursive(args)

        # Core Execution Logic
        def execute(m, f, a):
            if m == "builtins":
                if f == "getitem":
                    if isinstance(a, list) and len(a) >= 2: return a[0][a[1]]
                    if isinstance(a, dict): return a.get('obj', list(a.values())[0])[a.get('idx', 0)]
                    return a[0] if isinstance(a, list) else a
                func = __builtins__.get(f)
                if not func: raise ValueError(f"Builtin {f} not found.")
                if isinstance(a, dict): return func(**a)
                if isinstance(a, list): return func(*a)
                return func(a)
            
            mod = importlib.import_module(m)
            func = getattr(mod, f)
            
            if a is None: return func()
            if isinstance(a, dict):
                try:
                    sig = inspect.signature(func)
                    filtered = {k: v for k, v in a.items() if k in sig.parameters}
                    return func(**filtered)
                except: return func(**a)
            if isinstance(a, list): return func(*a)
            return func(a)

        result = execute(module, function, args)
        
        # Agnostic Serialization for HPM
        if isinstance(result, np.ndarray): result = result.tolist()
        elif isinstance(result, (sp.Basic, sp.Matrix)):
            try: result = float(result)
            except: result = str(result)
        elif not isinstance(result, (int, float, str, bool, list, dict, type(None))):
            result = str(result)
            
        return {"result": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def register_python_substrate():
    ToolRegistry.register(
        name="list_modules",
        tool_fn=list_modules,
        input_keys=["filter_list"],
        output_key="modules",
        cost=0.01,
        description="Return a curated list of top-level modules for exploration."
    )
    ToolRegistry.register(
        name="list_functions",
        tool_fn=list_functions,
        input_keys=["module"],
        output_key="functions",
        cost=0.01,
        description="Dynamically discover all callable functions in a module."
    )
    ToolRegistry.register(
        name="python_call",
        tool_fn=python_call,
        input_keys=["module", "function", "args"],
        output_key="result",
        cost=0.05,
        description="Execute a Python function with agnostic argument mapping."
    )
    print("[PythonSubstrate] Registered unified Python discovery tools.")


# Automatic registration
if "ToolRegistry" in globals() and not ToolRegistry.get_tool_info("python_call"):
    register_python_substrate()
