from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
import os
import json
import numpy as np
from hpm_ai_v6.agents.dependency_relation_agent import DependencyRelationAgent

def make_mock_dep_nlp(sentences_to_parse):
    """sentences_to_parse: List[List[Tuple[text, pos_, dep_, head_text, head_head_text]]]"""
    def nlp_pipe(texts, **kwargs):
        for token_list in sentences_to_parse:
            doc = MagicMock()
            tokens = []
            for text, pos_, dep_, head_text, head_head_text in token_list:
                tok = MagicMock()
                tok.text = text
                tok.pos_ = pos_
                tok.dep_ = dep_
                head = MagicMock()
                head.text = head_text
                head_head = MagicMock()
                head_head.text = head_head_text
                head.head = head_head
                tok.head = head
                tok.lower_ = text.lower()
                tokens.append(tok)
            doc.__iter__ = MagicMock(return_value=iter(tokens))
            yield doc
    mock = MagicMock()
    mock.pipe = nlp_pipe
    return mock

# "Alice followed the rabbit."
SIMPLE_SVO_NLP = make_mock_dep_nlp([
    [("Alice", "PROPN", "nsubj", "followed", "followed"),
     ("followed", "VERB", "ROOT", "followed", "followed"),
     ("the", "DET", "det", "rabbit", "rabbit"),
     ("rabbit", "NOUN", "dobj", "followed", "followed"),
     (".", "PUNCT", "punct", "followed", "followed")]
])

PREP_NLP = make_mock_dep_nlp([
    # "She ran into the hole."
    # ran: ROOT; into: prep of ran; hole: pobj of into
    [("She", "PRON", "nsubj", "ran", "ran"),
     ("ran", "VERB", "ROOT", "ran", "ran"),
     ("into", "ADP", "prep", "ran", "ran"),
     ("the", "DET", "det", "hole", "hole"),
     ("hole", "NOUN", "pobj", "into", "ran")]
])

def test_has_patterns_list():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    assert isinstance(agent.patterns, list)

def test_get_weights_parallel_to_patterns():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    assert len(agent.get_weights()) == len(agent.patterns)

def test_svo_produces_subject_of_edge():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    names = {p.name for p in agent.patterns}
    assert any("subject_of" in n and "alice" in n.lower() for n in names)

def test_svo_produces_object_of_edge():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    names = {p.name for p in agent.patterns}
    assert any("object_of" in n and "rabbit" in n.lower() for n in names)

def test_patterns_are_dim1_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    assert all(p.dim == 1 for p in agent.patterns)

def test_source_and_target_are_word_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    for p in agent.patterns:
        assert p.source is not None and p.source.name.startswith("word_")
        assert p.target is not None and p.target.name.startswith("word_")

def test_paging_lookup_contains_word_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    lookup = agent._paging_lookup()
    assert any(k.startswith("word_") for k in lookup)
    assert "word_alice" in lookup

def test_empty_corpus_produces_no_patterns():
    nlp = make_mock_dep_nlp([])
    agent = DependencyRelationAgent(nlp=nlp)
    agent.learn_from_corpus([])
    assert agent.patterns == []

def test_prep_produces_prep_edge():
    agent = DependencyRelationAgent(nlp=PREP_NLP)
    agent.learn_from_corpus(["She ran into the hole."])
    names = {p.name for p in agent.patterns}
    # For "hole", dep="pobj", head="into", head.head="ran"
    # Resulting edge should be ran -> hole with prep_into
    assert any("prep_into" in n for n in names), f"Expected prep_into edge, got {names}"
    # Verify the edge connection
    target_edge = next(p for p in agent.patterns if "prep_into" in p.name)
    assert target_edge.source.name == "word_ran"
    assert target_edge.target.name == "word_hole"

def test_save_load_round_trip(tmp_path):
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    path = str(tmp_path / "dep.db")
    agent.save(path)
    agent2 = DependencyRelationAgent(nlp=make_mock_dep_nlp([]))
    agent2.load(path)
    assert len(agent2.patterns) == len(agent.patterns)
    assert len(agent2.get_weights()) == len(agent.get_weights())
    assert agent2.patterns[0].name == agent.patterns[0].name
    assert agent2.patterns[0].source.name == agent.patterns[0].source.name
    assert agent2.patterns[0].target.name == agent.patterns[0].target.name


def test_load_migrates_legacy_json_to_sqlite(tmp_path):
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    legacy_path = str(tmp_path / "dep.json")
    db_path = str(tmp_path / "dep.db")
    agent.save(legacy_path)

    agent2 = DependencyRelationAgent(nlp=make_mock_dep_nlp([]))
    agent2.load(db_path)
    assert len(agent2.patterns) == len(agent.patterns)
    assert os.path.exists(db_path)

def test_integration_with_reasoning_agent():
    from types import SimpleNamespace
    from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    reader = SimpleNamespace(agents={
        "dependency": agent, "word": None, "phrase": None,
        "contextual": None, "semantic": None, "char": None, "causal": None,
    })
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    ra.refresh()
    alice = ra._resolve_cell("alice")
    rabbit = ra._resolve_cell("rabbit")
    assert alice is not None
    assert rabbit is not None
    path = ra._beam_search_path(alice, rabbit)
    assert path is not None, "Expected alice -> followed -> rabbit path"
    # alice -[subject_of]-> followed, followed -[object_of]-> rabbit
    assert len(path) == 2
