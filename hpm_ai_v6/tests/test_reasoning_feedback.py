from types import SimpleNamespace

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


def test_reinforce_edge_adds_pattern_to_word_agent():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        alice = Cell(name="word_alice", dim=0, embedding=[1, 0, 0])
        rabbit = Cell(name="word_rabbit", dim=0, embedding=[0, 1, 0])
        initial_count = len(reader.word_agent.patterns)
        reader._reinforce_edge(alice, rabbit, 0.7)
        assert len(reader.word_agent.patterns) == initial_count + 1
        assert reader.reasoning_agent._dirty is True


def test_reinforce_edge_skips_dim_mismatch():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        small = Cell(name="word_small", dim=0, embedding=[1, 0])
        large = Cell(name="word_large", dim=0, embedding=[0, 1, 0, 0, 0])
        initial_count = len(reader.word_agent.patterns)
        reader._reinforce_edge(small, large, 0.7)
        assert len(reader.word_agent.patterns) == initial_count


def test_maintenance_cycle_accepts_query_batch():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit down the hole.\n" * 5)
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        report = reader.maintenance_cycle(
            sentences=["Alice followed the rabbit."],
            query_batch=["Why did Alice follow the rabbit?"],
        )
        assert isinstance(report, dict)


def test_prioritize_sentences_uses_focus_words():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\nThe book was on the table.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        reader._focus_words = {"rabbit"}
        ordered = reader._prioritize_sentences([
            "The book was on the table.",
            "Alice followed the rabbit.",
        ])
        assert ordered[0] == "Alice followed the rabbit."


def test_reflect_then_reinforce_included_in_refresh():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit down the hole.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        alice = Cell(name="word_alice", dim=0, embedding=[1, 0, 0])
        rabbit = Cell(name="word_rabbit", dim=0, embedding=[0, 1, 0])
        hole = Cell(name="word_hole", dim=0, embedding=[0, 0, 1])
        edge_one = Cell(
            name="w_alice->rabbit",
            dim=1,
            embedding=(rabbit.as_numpy() - alice.as_numpy()).tolist(),
            source=alice,
            target=rabbit,
            weight=0.8,
        )
        edge_two = Cell(
            name="w_rabbit->hole",
            dim=1,
            embedding=(hole.as_numpy() - rabbit.as_numpy()).tolist(),
            source=rabbit,
            target=hole,
            weight=0.75,
        )
        reader.word_agent.patterns = [edge_one, edge_two]
        reader.word_agent.word_cells = {"alice": alice, "rabbit": rabbit, "hole": hole}
        reader.reasoning_agent.invalidate()

        signal = reader.reasoning_agent.reflect(["How does alice connect to hole?"])
        assert signal.derived_edges

        for src, tgt, score in signal.derived_edges:
            reader._reinforce_edge(src, tgt, score)

        reader.reasoning_agent._ensure_fresh()
        source_key = reader.reasoning_agent._cell_key(alice)
        targets = {
            edge.target_key
            for edge in reader.reasoning_agent._edge_index.get(source_key, [])
        }
        assert reader.reasoning_agent._cell_key(hole) in targets


def test_second_reasoning_pass_finds_reinforced_edge():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit down the hole.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        alice = Cell(name="word_alice", dim=0, embedding=[1, 0, 0])
        rabbit = Cell(name="word_rabbit", dim=0, embedding=[0, 1, 0])
        hole = Cell(name="word_hole", dim=0, embedding=[0, 0, 1])
        edge_one = Cell(
            name="w_alice->rabbit",
            dim=1,
            embedding=(rabbit.as_numpy() - alice.as_numpy()).tolist(),
            source=alice,
            target=rabbit,
            weight=0.8,
        )
        edge_two = Cell(
            name="w_rabbit->hole",
            dim=1,
            embedding=(hole.as_numpy() - rabbit.as_numpy()).tolist(),
            source=rabbit,
            target=hole,
            weight=0.75,
        )
        reader.word_agent.patterns = [edge_one, edge_two]
        reader.word_agent.word_cells = {"alice": alice, "rabbit": rabbit, "hole": hole}
        reader.reasoning_agent.invalidate()

        before = reader.reasoning_agent.reason_with_trace("How does alice connect to hole?")
        assert before["chosen_path"] is not None
        assert len(before["chosen_path"]["steps"]) == 2

        signal = reader.reasoning_agent.reflect(["How does alice connect to hole?"])
        for src, tgt, score in signal.derived_edges:
            reader._reinforce_edge(src, tgt, score)

        after = reader.reasoning_agent.reason_with_trace("How does alice connect to hole?")
        assert after["chosen_path"] is not None
        assert len(after["chosen_path"]["steps"]) == 1


def test_train_sequence_prunes_incompatible_word_patterns():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)

        alice = reader.word_agent._get_or_create_word_cell("alice")
        rabbit = reader.word_agent._get_or_create_word_cell("rabbit")
        bad_pattern = Cell(
            name="w_alice->rabbit",
            dim=1,
            embedding=[0.0] * 64,
            source=Cell(name="word_alice", dim=0, embedding=[0.0] * 64),
            target=Cell(name="word_rabbit", dim=0, embedding=[0.0] * 64),
        )
        reader.word_agent.patterns.append(bad_pattern)
        reader.word_agent._refresh_learner()

        reader.train_sequence(["alice rabbit"], enable_causal=False)

        repaired = [pattern for pattern in reader.word_agent.patterns if pattern.name == "w_alice->rabbit"]
        assert repaired
        assert all(len(pattern.as_numpy()) == len(alice.as_numpy()) for pattern in repaired)
        assert all(pattern.source is None or len(pattern.source.as_numpy()) == len(alice.as_numpy()) for pattern in repaired)
        assert all(pattern.target is None or len(pattern.target.as_numpy()) == len(rabbit.as_numpy()) for pattern in repaired)
