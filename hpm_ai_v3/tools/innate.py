"""
innate.py - Minimal viable innate tools for HPM agents.
Provides safe, general-purpose computational primitives.
"""

import re
import math
from typing import Dict, Any, List, Union

from hpm_ai_v3.tools.registry import ToolRegistry

# ----------------------------------------------------------------------
# Safe Arithmetic Evaluation
# ----------------------------------------------------------------------
try:
    from simpleeval import simple_eval
    SAFE_EVAL = True
except ImportError:
    SAFE_EVAL = False


def arithmetic_eval(expression: str) -> Dict[str, Any]:
    """Safely evaluate an arithmetic expression using simpleeval."""
    s_expr = str(expression).strip()
    
    # Discovery Robustness: Only attempt if it looks like math
    # Fix 3: Accept bare numbers. Must contain at least one digit.
    if not any(c.isdigit() for c in s_expr):
        return {"error": f"Expression '{s_expr}' does not appear to be arithmetic.", "status": "failed"}

    if not SAFE_EVAL:
        # Fallback to basic string parsing for simple cases if simpleeval missing
        try:
            # Very limited fallback for basic ops
            import ast
            import operator as op
            operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                         ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
                         ast.USub: op.neg}
            def eval_expr(node):
                if isinstance(node, ast.Num): return node.n
                elif isinstance(node, ast.BinOp):
                    return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return operators[type(node.op)](eval_expr(node.operand))
                else: raise TypeError(node)
            result = eval_expr(ast.parse(str(expression), mode='eval').body)
            return {"result": result, "status": "success"}
        except Exception as e:
            return {"error": f"simpleeval missing and fallback failed: {str(e)}", "status": "failed"}
    try:
        result = simple_eval(str(expression))
        return {"result": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# Type Conversion
# ----------------------------------------------------------------------
def to_float(x: Any) -> Dict[str, Any]:
    """Convert input to float."""
    try:
        if isinstance(x, list) and len(x) > 0: x = x[0]
        return {"result": float(x), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def to_int(x: Any) -> Dict[str, Any]:
    """Convert input to integer."""
    try:
        if isinstance(x, list) and len(x) > 0: x = x[0]
        return {"result": int(float(x)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def to_str(x: Any) -> Dict[str, Any]:
    """Convert input to string."""
    try:
        return {"result": str(x), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# String Operations
# ----------------------------------------------------------------------
def string_split(s: str, sep: str = None) -> Dict[str, Any]:
    """Split a string by separator (default: whitespace)."""
    try:
        if sep is None:
            return {"result": str(s).split(), "status": "success"}
        return {"result": str(s).split(str(sep)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def string_strip(s: str) -> Dict[str, Any]:
    """Remove leading/trailing whitespace."""
    try:
        return {"result": str(s).strip(), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def string_lower(s: str) -> Dict[str, Any]:
    """Convert string to lowercase."""
    try:
        return {"result": str(s).lower(), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def string_upper(s: str) -> Dict[str, Any]:
    """Convert string to uppercase."""
    try:
        return {"result": str(s).upper(), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def string_replace(s: str, old: str, new: str) -> Dict[str, Any]:
    """Replace occurrences of old with new in string."""
    try:
        return {"result": str(s).replace(str(old), str(new)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def string_contains(s: str, substring: str) -> Dict[str, Any]:
    """Check if string contains substring."""
    try:
        return {"result": str(substring) in str(s), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# List/Sequence Operations
# ----------------------------------------------------------------------
def list_length(obj: Any) -> Dict[str, Any]:
    """Return length of a list or string."""
    try:
        return {"result": len(obj), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def list_index(obj: Union[List, str], idx: int) -> Dict[str, Any]:
    """Get item at index from list or string."""
    try:
        i = int(idx)
        return {"result": obj[i], "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def list_slice(obj: Union[List, str], start: int = None, end: int = None) -> Dict[str, Any]:
    """Slice a list or string."""
    try:
        s = int(start) if start is not None else None
        e = int(end) if end is not None else None
        return {"result": obj[s:e], "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# Regex Pattern Matching
# ----------------------------------------------------------------------
def regex_findall(pattern: str, string: str) -> Dict[str, Any]:
    """Find all non-overlapping matches of pattern in string."""
    try:
        matches = re.findall(str(pattern), str(string))
        return {"result": matches, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def regex_search(pattern: str, string: str) -> Dict[str, Any]:
    """Search for first match of pattern in string."""
    try:
        match = re.search(str(pattern), str(string))
        if match:
            return {"result": match.group(), "span": match.span(), "status": "success"}
        return {"result": None, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def regex_sub(pattern: str, repl: str, string: str) -> Dict[str, Any]:
    """Replace occurrences of pattern with repl."""
    try:
        result = re.sub(str(pattern), str(repl), str(string))
        return {"result": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# Math Functions (from math module)
# ----------------------------------------------------------------------
def math_sin(x: float) -> Dict[str, Any]:
    try:
        return {"result": math.sin(float(x)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def math_cos(x: float) -> Dict[str, Any]:
    try:
        return {"result": math.cos(float(x)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def math_sqrt(x: float) -> Dict[str, Any]:
    try:
        return {"result": math.sqrt(float(x)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def math_pow(x: float, y: float) -> Dict[str, Any]:
    try:
        return {"result": math.pow(float(x), float(y)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def math_log(x: float, base: float = math.e) -> Dict[str, Any]:
    try:
        return {"result": math.log(float(x), float(base)), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def register_innate_tools():
    """Register the minimal viable innate tools for HPM agents."""
    
    # Arithmetic
    ToolRegistry.register("arithmetic", arithmetic_eval, ["expression"], "result", 0.01,
                          "Safely evaluate arithmetic expression.")
    
    # Type conversion
    ToolRegistry.register("float", to_float, ["x"], "result", 0.001,
                          "Convert input to float.")
    ToolRegistry.register("int", to_int, ["x"], "result", 0.001,
                          "Convert input to integer.")
    ToolRegistry.register("str", to_str, ["x"], "result", 0.001,
                          "Convert input to string.")
    
    # String Operations
    ToolRegistry.register("split", string_split, ["s", "sep"], "result", 0.005,
                          "Split string by separator.",
                          module="hpm_ai_v3.tools.innate", function="string_split")
    ToolRegistry.register("strip", string_strip, ["s"], "result", 0.001,
                          "Strip whitespace from string.",
                          module="hpm_ai_v3.tools.innate", function="string_strip")
    ToolRegistry.register("lower", string_lower, ["s"], "result", 0.001,
                          "Convert string to lowercase.")
    ToolRegistry.register("upper", string_upper, ["s"], "result", 0.001,
                          "Convert string to uppercase.")
    ToolRegistry.register("replace", string_replace, ["s", "old", "new"], "result", 0.002,
                          "Replace substring in string.")
    ToolRegistry.register("contains", string_contains, ["s", "substring"], "result", 0.001,
                          "Check if string contains substring.")

    # List operations
    ToolRegistry.register("len", list_length, ["obj"], "result", 0.001,
                          "Get length of list or string.")
    ToolRegistry.register("index", list_index, ["obj", "idx"], "result", 0.001,
                          "Get item at index.",
                          module="hpm_ai_v3.tools.innate", function="list_index")
    ToolRegistry.register("slice", list_slice, ["obj", "start", "end"], "result", 0.002,
                          "Slice a list or string.")

    # Regex
    ToolRegistry.register("re_findall", regex_findall, ["pattern", "string"], "result", 0.005,
                          "Find all regex matches in string.",
                          module="hpm_ai_v3.tools.innate", function="regex_findall")

    # Innate Substrate Extensions (Groups A-G)
    from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate
    substrate = InnateCognitiveSubstrate()
    ToolRegistry.register("get_type", substrate.get_type, ["value"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="get_type")
    ToolRegistry.register("describe_value", substrate.describe_value, ["value"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="describe_value")
    ToolRegistry.register("coerce", substrate.coerce, ["value", "target_type"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="coerce")
    ToolRegistry.register("extract_numbers", substrate.extract_numbers, ["text"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="extract_numbers")
    ToolRegistry.register("match_regex", substrate.match_regex, ["pattern", "text"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="match_regex")
    ToolRegistry.register("split_text", substrate.split_text, ["text", "delimiter"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="split_text")
    ToolRegistry.register("build_mapping", substrate.build_mapping, ["keys", "values"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="build_mapping")
    ToolRegistry.register("invert_mapping", substrate.invert_mapping, ["mapping"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="invert_mapping")
    ToolRegistry.register("get_nested", substrate.get_nested, ["obj", "path"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="get_nested")
    ToolRegistry.register("flatten", substrate.flatten, ["nested"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="flatten")
    ToolRegistry.register("group_by", substrate.group_by, ["items", "key_fn"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="group_by")
    ToolRegistry.register("normalize", substrate.normalize, ["values"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="normalize")
    ToolRegistry.register("entropy", substrate.entropy, ["probs"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="entropy")
    ToolRegistry.register("argmax", substrate.argmax, ["values"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="argmax")
    ToolRegistry.register("clamp", substrate.clamp, ["value", "lo", "hi"], "result", 0.001,
                          module="hpm_ai_v3.tools.innate_substrate", function="clamp")
    ToolRegistry.register("decompose_text", substrate.decompose_text, ["text"], "result", 0.01,
                          module="hpm_ai_v3.tools.innate_substrate", function="decompose_text")
    ToolRegistry.register("check_constraint", substrate.check_constraint, ["value", "constraint_str"], "result", 0.005,
                          module="hpm_ai_v3.tools.innate_substrate", function="check_constraint")
    
    # Math functions
    ToolRegistry.register("sin", math_sin, ["x"], "result", 0.01,
                          "Sine of angle in radians.")
    ToolRegistry.register("cos", math_cos, ["x"], "result", 0.01,
                          "Cosine of angle in radians.")
    ToolRegistry.register("sqrt", math_sqrt, ["x"], "result", 0.01,
                          "Square root.")
    ToolRegistry.register("pow", math_pow, ["x", "y"], "result", 0.01,
                          "Raise x to power y.")
    ToolRegistry.register("log", math_log, ["x", "base"], "result", 0.01,
                          "Logarithm of x with given base.")
    
    print("[InnateTools] Registered minimal viable innate tools.")


# Auto-register on import
# register_innate_tools()
