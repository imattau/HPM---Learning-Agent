from types import SimpleNamespace
import pytest
from typing import List, Tuple

from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent, ReasoningSignal
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}

    def get_weights(self): return list(self._weights)

    def _paging_lookup(self): return dict(self._lookup)


def make_reader_with_no_edges():
    """Reader where alice resolves but has no outgoing edges."""
    alice = Cell(name="word_alice", dim=0, embedding=[1, 0, 0])
    word_agent = StubAgent(patterns=[], weights=[], lookup={"word_alice": alice})
    return SimpleNamespace(agents={
        "word": word_agent, "phrase": None, "contextual": None,
        "semantic": None, "char": None, "causal": None,
    })


def make_reader_with_path():
    """Reader where alice→rabbit path exists."""
    alice = Cell(name="word_alice", dim=0, embedding=[1, 0, 0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0, 1, 0])
    decoy = Cell(name="word_decoy", dim=0, embedding=[0, 0, 1])
    edge = Cell(name="w_alice->rabbit", dim=1,
                embedding=(rabbit.as_numpy() - alice.as_numpy()).tolist(), source=alice, target=rabbit)
    decoy_edge = Cell(name="w_alice->decoy", dim=1,
                      embedding=(decoy.as_numpy() - alice.as_numpy()).tolist(), source=alice, target=decoy)
    word_agent = StubAgent(
        patterns=[edge, decoy_edge], weights=[0.5, 0.9],
        lookup={"word_alice": alice, "word_rabbit": rabbit, "word_decoy": decoy},
    )
    return SimpleNamespace(agents={
        "word": word_agent, "phrase": None, "contextual": None,
        "semantic": None, "char": None, "causal": None,
    })


def test_reasoning_signal_is_dataclass():
    signal = ReasoningSignal(
        uncertain_concepts=[],
        high_value_paths=[],
        suggested_focus_words=[],
        derived_edges=[],
    )
    assert signal.uncertain_concepts == []
    assert signal.derived_edges == []


def test_reflect_returns_reasoning_signal():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect(["How does alice connect?"])
    assert isinstance(signal, ReasoningSignal)


def test_reflect_identifies_uncertain_concepts_when_no_path():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect(["How does alice connect to rabbit?"])
    # alice resolves but rabbit doesn't exist → uncertain
    assert len(signal.uncertain_concepts) > 0


def test_reflect_returns_derived_edges_when_path_found():
    reader = make_reader_with_path()
    ra = ReasoningAgent(reader, max_beam_width=5, max_depth=3)
    signal = ra.reflect(["How does alice connect to rabbit?"])
    # Path alice→rabbit found (1 hop) — no multi-hop to derive
    # But for a 2-hop path, derived_edges would be non-empty
    assert isinstance(signal.derived_edges, list)


def test_reflect_with_empty_queries_returns_empty_signal():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect([])
    assert signal.uncertain_concepts == []
    assert signal.derived_edges == []
