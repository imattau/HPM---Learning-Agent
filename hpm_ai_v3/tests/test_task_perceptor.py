import pytest
from hpm_ai_v3.tools.task_perceptor import TaskPerceptor

@pytest.fixture
def p():
    return TaskPerceptor()

def test_numeric_type(p):
    percept = p.perceive("42.5")
    assert percept["input_type"] == "numeric"

def test_expression_type(p):
    percept = p.perceive("3 + 4 * 2")
    assert percept["input_type"] == "expression"

def test_string_type(p):
    percept = p.perceive("Is 'hello' a palindrome?")
    assert percept["input_type"] == "string"

def test_boolean_type(p):
    percept = p.perceive("Is 7 greater than 5?")
    assert percept["input_type"] == "boolean"

def test_list_type(p):
    percept = p.perceive("Sort the list [3, 1, 2]")
    assert percept["input_type"] == "list"

def test_mixed_type(p):
    percept = p.perceive("The temperature is 98.6 degrees")
    assert percept["input_type"] == "mixed"

def test_tokens_lowercase(p):
    percept = p.perceive("Hello World")
    assert "hello" in percept["tokens"]
    assert "world" in percept["tokens"]

def test_bare_number_numeric(p):
    assert p.perceive("42")["input_type"] == "numeric"

def test_expression_with_spaces(p):
    assert p.perceive("10 + 20 - 3")["input_type"] == "expression"

def test_operation_compute(p):
    assert p.perceive("What is 5 + 3?")["operation"] == "compute"

def test_operation_classify(p):
    assert p.perceive("Determine the type of 42")["operation"] == "classify"

def test_operation_extract(p):
    assert p.perceive("Count the words in this sentence")["operation"] == "extract"

def test_operation_compare(p):
    assert p.perceive("Is 7 greater than 5?")["operation"] == "compare"

def test_operation_transform(p):
    assert p.perceive("Reverse the string hello")["operation"] == "transform"

def test_numeric_values_extracted(p):
    percept = p.perceive("Add 10 and 20")
    assert 10.0 in percept["numeric_values"]
    assert 20.0 in percept["numeric_values"]

def test_is_question_true(p):
    assert p.perceive("Is 3 prime?")["is_question"] is True

from unittest.mock import MagicMock
import numpy as np
from hpm_ai_v3.tools.tool_selector import ToolSelector

def _make_pattern(tool_name):
    pat = MagicMock()
    pat.tool_name = tool_name
    pat.module = None
    pat.function = None
    pat.action_type = tool_name
    pat.weight = 1.0
    return pat

def _make_selector():
    lm = MagicMock()
    lm._embed.return_value = [0.1] * 128
    return ToolSelector(lm=lm, alpha=1.0)

def test_filter_numeric_blocks_split():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split"), _make_pattern("float")]
    weights = np.array([1.0, 1.0, 1.0])
    percept = {"input_type": "numeric", "operation": "compute",
               "numeric_values": [42.0], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] > 0.0   # arithmetic allowed
    assert result[1] == 0.0  # split zeroed
    assert result[2] > 0.0   # float allowed

def test_filter_string_blocks_arithmetic():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split"), _make_pattern("re_findall")]
    weights = np.array([1.0, 1.0, 1.0])
    percept = {"input_type": "string", "operation": "transform",
               "numeric_values": [], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] == 0.0  # arithmetic zeroed
    assert result[1] > 0.0   # split allowed
    assert result[2] > 0.0   # re_findall allowed

def test_filter_mixed_allows_all():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split")]
    weights = np.array([1.0, 1.0])
    percept = {"input_type": "mixed", "operation": "compute",
               "numeric_values": [], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] > 0.0
    assert result[1] > 0.0

from hpm_ai_v3.tools.innate import arithmetic_eval

def test_arithmetic_bare_float():
    result = arithmetic_eval("42.5")
    assert result["status"] == "success"
    assert result["result"] == 42.5

def test_arithmetic_bare_int():
    result = arithmetic_eval("10")
    assert result["status"] == "success"
    assert result["result"] == 10.0

def test_arithmetic_bare_negative():
    result = arithmetic_eval("-7")
    assert result["status"] == "success"
    assert result["result"] == -7.0

def test_arithmetic_expression_still_works():
    result = arithmetic_eval("3 + 4")
    assert result["status"] == "success"
    assert result["result"] == 7

def test_arithmetic_rejects_pure_text():
    result = arithmetic_eval("hello world")
    assert result["status"] == "failed"

def test_perceptor_standalone_all_fields():
    """All percept fields present, correctly typed, and values in valid sets."""
    perceptor = TaskPerceptor()
    percept = perceptor.perceive("What is 3 + 4?")
    assert isinstance(percept["input_type"], str)
    assert isinstance(percept["operation"], str)
    assert isinstance(percept["numeric_values"], list)
    assert isinstance(percept["tokens"], list)
    assert isinstance(percept["is_question"], bool)
    valid_types = {"numeric", "expression", "string", "boolean", "list", "mixed"}
    valid_ops = {"compute", "classify", "extract", "compare", "transform"}
    assert percept["input_type"] in valid_types
    assert percept["operation"] in valid_ops
    assert percept["is_question"] is True  # "What is 3 + 4?" ends with "?"
