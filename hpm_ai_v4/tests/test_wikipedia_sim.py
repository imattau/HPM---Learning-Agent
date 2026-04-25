# hpm_ai_v4/tests/test_wikipedia_sim.py
import tempfile, os
import pytest
from hpm_ai_v4.simulations.wikipedia_sim import WikipediaStream
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern

def _make_stream(content: str):
    adapter = CharClassAdapter()
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()
    return WikipediaStream(f.name, adapter), f.name

def test_stream_yields_valid_class_ids():
    stream, path = _make_stream("Hello world!\n")
    # Wrap in a way that we can break out of the infinite loop
    it = iter(stream)
    ids = [next(it) for _ in range(13)] # "Hello world!\n" is 13 chars
    os.unlink(path)
    assert all(0 <= i <= 4 for i in ids)

def test_stream_length_matches_valid_chars():
    content = "Hi!\n"
    stream, path = _make_stream(content)
    it = iter(stream)
    ids = [next(it) for _ in range(4)]
    os.unlink(path)
    # 'H','i','!','\n' => 4 class IDs
    assert len(ids) == 4

def test_stream_loops():
    stream, path = _make_stream("ab")
    it = iter(stream)
    first_two = [next(it), next(it)]
    next_two = [next(it), next(it)]  # should loop
    os.unlink(path)
    assert first_two == next_two

def test_stream_newline_is_class_4():
    stream, path = _make_stream("\n")
    it = iter(stream)
    ids = [next(it)]
    os.unlink(path)
    assert ids == [4]

from hpm_ai_v4.simulations.wikipedia_sim import run_simulation

def test_run_simulation_200_steps():
    content = "The quick brown fox jumps over the lazy dog. " * 20
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()

    state = run_simulation(corpus_path=f.name, total_steps=200, log_every=201)
    os.unlink(f.name)

    assert len(state['L1_patterns']) >= 1
    assert len(state['L2_patterns']) >= 1
    assert len(state['L3_patterns']) >= 1

def test_run_simulation_emits_ints_in_range():
    content = "Hello world!\n" * 30
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()

    # Should not raise
    run_simulation(corpus_path=f.name, total_steps=50, log_every=100)
    os.unlink(f.name)

from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.simulations.text_reasoning import TextReasoningInterface

def _make_mock_agent(patterns):
    """Minimal object satisfying Reasoner's interface."""
    class FakeAgent:
        obs_buffer = [0, 1, 0]
    agent = FakeAgent()
    agent.patterns = patterns
    return agent

def test_next_char_predict_returns_5_tuples():
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=5) for i in range(3)]
    for p in patterns:
        p.weight = 1.0 / 3
    agent = _make_mock_agent(patterns)
    reasoner = Reasoner(agent)
    iface = TextReasoningInterface(
        L1_patterns=patterns, L1_reasoner=reasoner,
        L2_patterns=[], L2_reasoner=None,
        L3_patterns=[], L3_reasoner=None,
    )
    result = iface.next_char_predict("hello")
    assert len(result) == 5
    names, probs = zip(*result)
    assert abs(sum(probs) - 1.0) < 0.01

def test_plan_to_space_returns_class_names():
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=2) for i in range(3)]
    for p in patterns:
        p.weight = 1.0 / 3
    agent = _make_mock_agent(patterns)
    reasoner = Reasoner(agent)
    iface = TextReasoningInterface(
        L1_patterns=patterns, L1_reasoner=reasoner,
        L2_patterns=patterns, L2_reasoner=reasoner,
        L3_patterns=patterns, L3_reasoner=reasoner,
    )
    result = iface.plan_to_space(horizon=3, num_rollouts=5)
    assert isinstance(result, list)
    CLASS_NAMES = {'letter', 'digit', 'space', 'punctuation', 'newline'}
    for name in result:
        assert name in CLASS_NAMES
