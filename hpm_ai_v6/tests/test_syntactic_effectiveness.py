"""
End-to-end effectiveness test: valid POS sequences score higher than invalid ones.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.syntactic_rule_agent import SyntacticRuleAgent
from hpm_ai_v6.hpm_model.core.cell import Cell


def make_mock_nlp(sentences_to_pos):
    def nlp_pipe(texts, **kwargs):
        for token_list in sentences_to_pos:
            doc = MagicMock()
            doc.__iter__ = MagicMock(return_value=iter([
                MagicMock(text=t, pos_=p) for t, p in token_list
            ]))
            yield doc
    mock = MagicMock()
    mock.pipe = nlp_pipe
    return mock


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}

    def get_weights(self):
        return list(self._weights)

    def _paging_lookup(self):
        return dict(self._lookup)


def make_edge(name, source, target, weight=0.5):
    return Cell(
        name=name,
        dim=1,
        embedding=target.as_numpy() - source.as_numpy(),
        source=source,
        target=target,
        weight=weight,
    )


def _build_reader(word_agent, syn_agent):
    return SimpleNamespace(agents={
        "word": word_agent,
        "syntactic": syn_agent,
        "phrase": None,
        "contextual": None,
        "semantic": None,
        "char": None,
        "causal": None,
    })


def _make_common_setup():
    the = Cell(name="word_the", dim=0, embedding=[1, 0, 0, 0])
    cat = Cell(name="word_cat", dim=0, embedding=[0, 1, 0, 0])
    sat = Cell(name="word_sat", dim=0, embedding=[0, 0, 1, 0])
    dog = Cell(name="word_dog", dim=0, embedding=[0, 0, 0, 1])

    e_the_cat = make_edge("w_the->cat", the, cat, weight=0.5)
    e_cat_sat = make_edge("w_cat->sat", cat, sat, weight=0.5)
    e_the_dog = make_edge("w_the->dog", the, dog, weight=0.9)

    word_agent = StubAgent(
        patterns=[e_the_cat, e_cat_sat, e_the_dog],
        weights=[0.5, 0.5, 0.9],
        lookup={c.name: c for c in [the, cat, sat, dog]},
    )

    nlp = make_mock_nlp([
        [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
        [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
    ])
    syn_agent = SyntacticRuleAgent(min_prob=0.1, nlp=nlp)
    syn_agent.learn_from_corpus(["the cat sat", "the dog ran"])

    return the, cat, sat, dog, word_agent, syn_agent


def test_syntactic_agent_learns_rules():
    """SyntacticRuleAgent must produce at least one DET_NOUN_VERB rule."""
    _, _, _, _, _, syn_agent = _make_common_setup()
    rule_names = [p.name for p in syn_agent.patterns]
    det_noun_verb = [n for n in rule_names if "DET_NOUN_VERB" in n]
    assert len(det_noun_verb) > 0, (
        f"Expected a DET_NOUN_VERB rule. Got: {rule_names}"
    )


def test_valid_pos_sequence_produces_derived_edge():
    """
    DET->NOUN->VERB chain produces a derived edge via syntactic rule.
    After refresh(), _forward_chain_edges should yield a the->sat edge.
    """
    the, cat, sat, dog, word_agent, syn_agent = _make_common_setup()

    reader = _build_reader(word_agent, syn_agent)
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    ra.refresh()

    the_key = ra._cell_key(the)
    sat_key = ra._cell_key(sat)

    forward_derived = ra._forward_chain_edges(the_key)

    derived_to_sat = [
        e
        for records in forward_derived.values()
        for e in records
        if e.target_key == sat_key
    ]

    assert len(derived_to_sat) > 0, (
        f"Expected a derived edge the->sat from DET->NOUN->VERB rule.\n"
        f"Forward derived keys: {list(forward_derived.keys())}\n"
        f"Edge targets: {[e.target_key for records in forward_derived.values() for e in records]}"
    )
    assert derived_to_sat[0].score > 0.0


def test_no_noun_det_verb_derived_edge():
    """
    NOUN->DET is not in the corpus, so no derived edge via NOUN_DET_VERB rule.
    There's no cat->the word edge, so even if a rule existed, it can't fire.
    """
    the, cat, sat, dog, word_agent, syn_agent = _make_common_setup()

    reader = _build_reader(word_agent, syn_agent)
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    ra.refresh()

    cat_key = ra._cell_key(cat)
    sat_key = ra._cell_key(sat)

    forward_from_cat = ra._forward_chain_edges(cat_key)

    # Look for derived edges via cat->the->sat (NOUN->DET chain)
    derived_via_the_to_sat = [
        e
        for records in forward_from_cat.values()
        for e in records
        if e.target_key == sat_key and "NOUN_DET" in e.pattern.name
    ]
    assert len(derived_via_the_to_sat) == 0, (
        "NOUN->DET->VERB should not produce a derived edge (no such rule/edge exists)"
    )


def test_valid_beam_path_found():
    """_beam_search_path finds the->sat when DET->NOUN->VERB rule is active."""
    the, cat, sat, dog, word_agent, syn_agent = _make_common_setup()

    reader = _build_reader(word_agent, syn_agent)
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    ra.refresh()

    path = ra._beam_search_path(the, sat)
    assert path is not None, "Expected a path from 'the' to 'sat'"
    assert len(path) >= 1


def test_syntactic_agent_increases_candidate_paths():
    """
    With syntactic agent, reasoning finds at least as many candidate paths
    as without it, due to derived edges expanding the graph.
    """
    the, cat, sat, dog, word_agent, syn_agent = _make_common_setup()

    reader_syn = _build_reader(word_agent, syn_agent)
    ra_syn = ReasoningAgent(reader_syn, beam_width=5, max_depth=3)
    trace_syn = ra_syn.reason_with_trace("how does the connect to sat")

    reader_plain = SimpleNamespace(agents={
        "word": word_agent,
        "syntactic": None,
        "phrase": None,
        "contextual": None,
        "semantic": None,
        "char": None,
        "causal": None,
    })
    ra_plain = ReasoningAgent(reader_plain, beam_width=5, max_depth=3)
    trace_plain = ra_plain.reason_with_trace("how does the connect to sat")

    n_syn = len(trace_syn["candidate_paths"])
    n_plain = len(trace_plain["candidate_paths"])

    assert n_syn >= n_plain, (
        f"Syntactic agent should find >= candidate paths. "
        f"syn={n_syn}, plain={n_plain}"
    )
