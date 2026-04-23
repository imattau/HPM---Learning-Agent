"""Test InnateCognitiveSubstrate.find_in_list and list_index alias routing."""
import pytest
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate


@pytest.fixture
def innate():
    return InnateCognitiveSubstrate()


def test_find_in_list_basic(innate):
    """find_in_list(3, [1,2,3]) returns index 2."""
    assert innate.find_in_list(3, [1, 2, 3]) == 2


def test_find_in_list_not_found(innate):
    """find_in_list returns None when value is absent."""
    assert innate.find_in_list(99, [1, 2, 3]) is None


def test_find_in_list_first_occurrence(innate):
    """find_in_list returns index of first occurrence."""
    assert innate.find_in_list(2, [2, 2, 2]) == 0


def test_list_index_alias_exists(innate):
    """list_index must be an accessible alias for find_in_list."""
    assert hasattr(innate, "list_index"), "InnateCognitiveSubstrate missing list_index alias"


def test_list_index_alias_works(innate):
    """list_index(3, [1, 2, 3]) returns same result as find_in_list."""
    assert innate.list_index(3, [1, 2, 3]) == 2


def test_resolve_call_find_in_list(innate):
    """resolve_call resolves both required args for find_in_list."""
    pool = [3, [1, 2, 3, 4]]
    module = "hpm_ai_v3.tools.innate_substrate"
    mod_out, fn_out, args = innate.resolve_call(module, "find_in_list", pool, "find 3 in list")
    assert fn_out == "find_in_list"
    # Must have exactly 2 args resolved
    assert len(args) == 2, f"Expected 2 args, got {len(args)}: {args}"


def test_list_index_missing_arg_raises_type_error(innate):
    """Calling list_index with only one arg raises TypeError with useful message."""
    # Note: list_index(3) will raise TypeError due to missing 'lst'
    with pytest.raises(TypeError):
        innate.list_index(3)
