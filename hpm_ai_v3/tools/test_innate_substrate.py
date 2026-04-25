# hpm_ai_v3/tools/test_innate_substrate.py
import pytest
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate

@pytest.fixture
def substrate():
    return InnateCognitiveSubstrate()

def test_inspect_signature_math_sqrt(substrate):
    sig = substrate.inspect_signature("math", "sqrt")
    assert sig is not None
    assert any(p["name"] == "x" for p in sig["params"])

def test_inspect_signature_re_findall(substrate):
    sig = substrate.inspect_signature("re", "findall")
    assert sig is not None
    assert len(sig["params"]) >= 2

def test_get_type_int(substrate):
    assert substrate.get_type(42) == "int"

def test_get_type_float(substrate):
    assert substrate.get_type(3.14) == "float"

def test_get_type_str(substrate):
    assert substrate.get_type("hello") == "str"

def test_get_type_list(substrate):
    assert substrate.get_type([1, 2]) == "list"

def test_describe_value_numeric_str(substrate):
    d = substrate.describe_value("42")
    assert d["type"] == "str"
    assert d["numeric_value"] == 42.0

def test_describe_value_list(substrate):
    d = substrate.describe_value([1, 2, 3])
    assert d["length"] == 3

def test_list_callable_functions(substrate):
    fns = substrate.list_callable_functions("math")
    names = [f["name"] for f in fns]
    assert "sqrt" in names
    assert "factorial" in names

# Group B tests
def test_coerce_str_to_int(substrate):
    assert substrate.coerce("42", "int") == 42

def test_coerce_str_to_float(substrate):
    assert substrate.coerce("3.14", "float") == 3.14

def test_coerce_returns_none_on_failure(substrate):
    assert substrate.coerce("hello", "int") is None

def test_to_int_from_float(substrate):
    assert substrate.to_int(3.9) == 3

def test_to_float_from_str(substrate):
    assert substrate.to_float("2.5") == 2.5

def test_to_str(substrate):
    assert substrate.to_str(42) == "42"

def test_to_list_from_str(substrate):
    result = substrate.to_list("hello")
    assert isinstance(result, list)

def test_safe_call_success(substrate):
    result = substrate.safe_call("math", "sqrt", 144)
    assert result == 12.0

def test_safe_call_error_returns_dict(substrate):
    result = substrate.safe_call("math", "sqrt", "not_a_number")
    assert isinstance(result, dict)
    assert "error" in result

# Group C tests
def test_extract_numbers_from_text(substrate):
    assert substrate.extract_numbers("The price is 42 and 7") == [42.0, 7.0]

def test_extract_numbers_empty(substrate):
    assert substrate.extract_numbers("no numbers here") == []

def test_match_regex(substrate):
    assert substrate.match_regex(r"\d+", "abc 42 def 7") == ["42", "7"]

def test_find_in_list(substrate):
    assert substrate.find_in_list(42, [1, 42, 3]) == 1

def test_find_in_list_missing(substrate):
    assert substrate.find_in_list(99, [1, 2, 3]) is None

def test_split_text_default(substrate):
    assert substrate.split_text("hello world") == ["hello", "world"]

def test_split_text_delimiter(substrate):
    assert substrate.split_text("a,b,c", ",") == ["a", "b", "c"]

def test_compare_values_greater(substrate):
    assert substrate.compare_values(10, 5) == "greater"

def test_compare_values_equal(substrate):
    assert substrate.compare_values(5, 5) == "equal"

def test_compare_values_incomparable(substrate):
    assert substrate.compare_values("hello", 5) == "incomparable"

# resolve_call tests
def test_resolve_call_math_sqrt(substrate):
    mod, fn, args = substrate.resolve_call("math", "sqrt", [144], "")
    assert mod == "math"
    assert fn == "sqrt"
    assert args == [144.0] or args == [144]

def test_resolve_call_extracts_from_text(substrate):
    mod, fn, args = substrate.resolve_call("math", "sqrt", [], "compute sqrt of 25")
    assert args[0] == 25.0

def test_resolve_call_re_findall(substrate):
    mod, fn, args = substrate.resolve_call("re", "findall", [], "The numbers are 42 and 7")
    assert mod == "re"
    assert fn == "findall"
    # first arg should be a pattern string, second the text
    assert len(args) >= 2

def test_substrate_resolves_math_sqrt_from_text():
    """End-to-end: substrate resolves math.sqrt args from task text."""
    from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate
    s = InnateCognitiveSubstrate()
    mod, fn, args = s.resolve_call("math", "sqrt", [], "compute sqrt of 144")
    assert mod == "math"
    result = s.safe_call(mod, fn, *args)
    assert result == 12.0
