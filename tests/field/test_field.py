import pytest
from hpm.field.field import PatternField


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
