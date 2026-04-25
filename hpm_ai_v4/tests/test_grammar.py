import pytest
from hpm_ai_v4.tools.grammar import NLTKGrammarLibrary

@pytest.fixture(scope="module")
def grammar():
    return NLTKGrammarLibrary(download=True)

def test_grammar_pos_tagging(grammar):
    assert grammar.get_pos("the") == "DT"
    assert grammar.get_pos("cat") == "NN"
    # "runs" falls back to NN in heuristic unless we add it
    assert grammar.get_pos("running") == "VB"

def test_grammar_transition_validation(grammar):
    # Determiner -> Noun is valid
    assert grammar.is_valid_transition("the", "cat")
    # Determiner -> Verb (VB) is invalid in heuristic transitions
    assert grammar.is_valid_transition("the", "running") == False

def test_grammar_sequence_scoring(grammar):
    good_seq = ["the", "cat", "running"] # DT, NN, VB
    bad_seq = ["the", "running", "the"] # DT, VB, DT (VB -> DT is allowed but DT -> VB is not)
    
    score_good = grammar.score_sequence(good_seq)
    score_bad = grammar.score_sequence(bad_seq)
    
    assert score_good > score_bad
    assert 0 <= score_good <= 1
    assert 0 <= score_bad <= 1

def test_empty_sequence(grammar):
    assert grammar.score_sequence([]) == 0.0
