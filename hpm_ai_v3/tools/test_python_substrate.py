import pytest
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.python_substrate import register_python_substrate

def setup_function():
    register_python_substrate()

def test_str_upper_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.upper", args=["hello world"])
    assert result.get("result") == "HELLO WORLD"

def test_str_lower_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.lower", args=["HELLO"])
    assert result.get("result") == "hello"

def test_str_split_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.split", args=["hello world"])
    assert result.get("result") == ["hello", "world"]
