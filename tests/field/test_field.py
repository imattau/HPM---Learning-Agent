import numpy as np
import pytest
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def _pattern():
    return GaussianPattern(np.zeros(2), np.eye(2))


def test_empty_field_returns_zero_freq():
    field = PatternField()
    assert field.freq("uuid-1") == pytest.approx(0.0)


def test_single_agent_single_pattern_freq_is_one():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.freq("uuid-1") == pytest.approx(1.0)


def test_two_agents_same_pattern_uuid_sums_weights():
    # Both agents have uuid-1: total mass = 1.0, freq = 1.0
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.5)])
    field.register("agent-2", [("uuid-1", 0.5)])
    assert field.freq("uuid-1") == pytest.approx(1.0)


def test_two_agents_different_patterns_split_freq():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.6)])
    field.register("agent-2", [("uuid-2", 0.4)])
    assert field.freq("uuid-1") == pytest.approx(0.6)
    assert field.freq("uuid-2") == pytest.approx(0.4)


def test_register_overwrites_previous_for_same_agent():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    field.register("agent-1", [("uuid-2", 1.0)])  # replaces previous
    assert field.freq("uuid-1") == pytest.approx(0.0)
    assert field.freq("uuid-2") == pytest.approx(1.0)


def test_unknown_pattern_id_returns_zero():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.freq("uuid-unknown") == pytest.approx(0.0)


def test_field_quality_empty_returns_zero_diversity():
    field = PatternField()
    quality = field.field_quality()
    assert quality["diversity"] == pytest.approx(0.0)
    assert quality["redundancy"] == pytest.approx(0.0)


def test_field_quality_two_equal_patterns_has_positive_diversity():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.5)])
    field.register("agent-2", [("uuid-2", 0.5)])
    quality = field.field_quality()
    assert quality["diversity"] > 0.0


def test_freqs_for_returns_list_matching_pattern_ids():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.6), ("uuid-2", 0.4)])
    freqs = field.freqs_for(["uuid-1", "uuid-2", "uuid-3"])
    assert freqs[0] == pytest.approx(0.6)
    assert freqs[1] == pytest.approx(0.4)
    assert freqs[2] == pytest.approx(0.0)


def test_n_agents_property():
    field = PatternField()
    assert field.n_agents == 0
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.n_agents == 1
    field.register("agent-2", [("uuid-2", 1.0)])
    assert field.n_agents == 2


def test_field_quality_two_equal_patterns_exact_entropy():
    import math
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.5)])
    field.register("agent-2", [("uuid-2", 0.5)])
    quality = field.field_quality()
    assert quality["diversity"] == pytest.approx(math.log(2))


def test_broadcast_appends_to_queue():
    field = PatternField()
    p = _pattern()
    field.broadcast('agent_a', p)
    queue = field.drain_broadcasts()
    assert len(queue) == 1
    assert queue[0][0] == 'agent_a'
    assert queue[0][1] is p


def test_drain_broadcasts_clears_queue():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    field.drain_broadcasts()
    assert field.drain_broadcasts() == []


def test_drain_broadcasts_returns_independent_list():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    result = field.drain_broadcasts()
    result.append(('extra', _pattern()))
    assert field.drain_broadcasts() == []


def test_multiple_broadcasts_accumulated():
    field = PatternField()
    field.broadcast('a', _pattern())
    field.broadcast('b', _pattern())
    queue = field.drain_broadcasts()
    assert len(queue) == 2


def test_broadcast_message_appends_to_separate_queue():
    field = PatternField()
    message = {"kind": "structural"}
    field.broadcast_message('agent_a', message)
    queue = field.drain_messages()
    assert len(queue) == 1
    assert queue[0] == ('agent_a', message)


def test_drain_messages_clears_queue_without_touching_pattern_broadcasts():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    field.broadcast_message('agent_b', {"kind": "structural"})
    messages = field.drain_messages()
    broadcasts = field.drain_broadcasts()
    assert messages == [('agent_b', {"kind": "structural"})]
    assert len(broadcasts) == 1


def test_drain_broadcasts_leaves_message_queue_intact():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    field.broadcast_message('agent_b', {"kind": "structural"})
    field.drain_broadcasts()
    assert field.drain_messages() == [('agent_b', {"kind": "structural"})]
