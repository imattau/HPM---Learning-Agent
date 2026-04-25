import pytest
import nltk
from hpm_ai_v4.tools.dictionary import NLTKWordList

@pytest.fixture(scope="module")
def dictionary():
    # Only load once for all tests in this module
    return NLTKWordList(download=True)

def test_dictionary_contains_common_words(dictionary):
    assert dictionary.contains("apple")
    assert dictionary.contains("science")
    assert dictionary.contains("pattern")
    # Case insensitivity
    assert dictionary.contains("APPLE")

def test_dictionary_does_not_contain_nonsense(dictionary):
    assert not dictionary.contains("qwertyuiop")
    assert not dictionary.contains("hpmframework")

def test_dictionary_prefix_match(dictionary):
    assert dictionary.is_prefix("app")
    assert dictionary.is_prefix("sci")
    assert not dictionary.is_prefix("xyzq")

def test_dictionary_completions(dictionary):
    comps = dictionary.completions("patt", max_suggestions=3)
    assert len(comps) <= 3
    assert any("pattern" in c for c in comps)

def test_dictionary_score_word(dictionary):
    assert dictionary.score_word("apple") == 1.0
    assert dictionary.score_word("nonword123") == 0.0

def test_empty_input_handling(dictionary):
    assert dictionary.is_prefix("") == True
    assert dictionary.completions("") == []
