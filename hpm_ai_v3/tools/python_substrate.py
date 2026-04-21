"""
python_substrate.py - Unified substrate for dynamic Python module exploration and execution.
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
    return {"modules": modules, "status": "success"}


def list_functions(module: str) -> Dict[str, Any]:
    """Dynamically discover all callable functions in a module with their signatures."""
    try:
        mod = importlib.import_module(module)
        functions = []
        
        for name, func in inspect.getmembers(mod, inspect.isroutine):
            if name.startswith("_"): continue
            
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
            
        return {"functions": sorted(functions, key=lambda x: x['name']), "module": module, "count": len(functions), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def python_call(module: str, function: str, args: Any = None) -> Dict[str, Any]:
    """
    Agnostic execution: import a module and call a function with arguments.
    Supports dict (keyword), list (positional), or single value.
    """
    import numpy as np
    import sympy as sp
    
    def try_float_recursive(v):
        if isinstance(v, str):
            # Try float first
            try: return float(v)
            except: 
                # Try sympify for math expressions
                if any(op in v for op in ["log", "sqrt", "*", "/", "+", "-"]):
                    try: return float(sp.sympify(v))
                    except: pass
                return v
        if isinstance(v, list): return [try_float_recursive(x) for x in v]
        if isinstance(v, dict): return {k: try_float_recursive(x) for k, x in v.items()}
        return v

    try:
        # 0. CHILD-LIKE ROBUSTNESS: Unbox single-element lists for common conversion functions
        if function in ["float", "int", "sympify", "abs", "round", "len", "sqrt", "log", "exp", "sin", "cos"]:
            if isinstance(args, list) and len(args) == 1:
                args = args[0]
            elif isinstance(args, dict) and len(args) == 1:
                args = list(args.values())[0]

        # 0.1 PRE-EMPTIVE TYPE ALIGNMENT for math/numpy/operators
        if module in ["math", "numpy", "operator"] or function in ["float", "int", "sympify"]:
            args = try_float_recursive(args)

        # 1. SPECIAL SUBSTRATES (NLP)
        if module == "spacy":
            import spacy
            try:
                if not hasattr(python_call, "_nlp"):
                    python_call._nlp = spacy.load("en_core_web_sm")
                nlp = python_call._nlp
            except:
                if function == "tokenize":
                    text = str(args[0]) if isinstance(args, list) else str(args)
                    return {"result": text.split(), "status": "success"}
                return {"error": "spacy model 'en_core_web_sm' not found", "status": "failed"}

            text = str(args[0]) if isinstance(args, list) else str(args)
            doc = nlp(text)
            if function == "tokenize": result = [t.text for t in doc]
            elif function == "pos_tags": result = [{"text": t.text, "pos": t.pos_} for t in doc]
            elif function == "nouns": result = [chunk.text for chunk in doc.noun_chunks]
            elif function == "entities": result = [{"text": e.text, "label": e.label_} for e in doc.ents]
            else:
                func = getattr(nlp, function, None) or getattr(doc, function)
                result = func()
            return {"result": result, "status": "success"}

        elif module == "textblob":
            from textblob import TextBlob
            text = str(args[0]) if isinstance(args, list) else str(args)
            blob = TextBlob(text)
            if function == "sentiment": result = {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}
            elif function == "noun_phrases": result = list(blob.noun_phrases)
            else:
                func = getattr(blob, function)
                result = func()
            return {"result": result, "status": "success"}

        # 2. CORE EXECUTION LOGIC
        def execute(m, f, a):
            # Handle instance methods like "str.upper", "str.split", "str.lower"
            if "." in f:
                method_name = f.split(".", 1)[1]
                obj = a[0] if isinstance(a, list) else a
                method_args = a[1:] if isinstance(a, list) else []
                return getattr(str(obj), method_name)(*method_args)

            if m == "builtins":
                if f == "getitem":
                    if isinstance(a, list) and len(a) >= 2: return a[0][a[1]]
                    if isinstance(a, dict): return a.get('obj', list(a.values())[0])[a.get('idx', 0)]
                    return a[0] if isinstance(a, list) else a
                func = __builtins__.get(f)
                if not func: raise ValueError(f"Builtin {f} not found.")
                # Builtins usually don't take keyword args like this, handle positional
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

        # RETRY LOOP FOR CHILD-LIKE ADAPTATION
        try:
            result = execute(module, function, args)
        except (TypeError, ValueError) as te:
            # Type error or Value error? Try one more level of conversion
            args = try_float_recursive(args)
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
