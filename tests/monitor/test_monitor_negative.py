import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.tiered_store import TieredStore
from hpm.field.field import PatternField
from hpm.monitor.structural_law import StructuralLawMonitor, _taboo_overlap
from hpm.monitor.recombination_strategist import RecombinationStrategist
from hpm.patterns.gaussian import GaussianPattern


def _make_agent(agent_id, gamma_neg=0.3):
    cfg = AgentConfig(agent_id=agent_id, feature_dim=4, gamma_neg=gamma_neg)
    store = TieredStore()
    agent = Agent(cfg, store=store)
    return agent


def _seed_negative(agent, mu_list):
    """Seed negative patterns directly into _tier2_negative."""
    for mu in mu_list:
        p = GaussianPattern(mu=np.array(mu, dtype=float), sigma=np.eye(4) * 0.1)
        agent.store._tier2_negative.save(p, 0.5, agent.agent_id)


# ── Test 1: negative_count in monitor report ──────────────────────────────────

def test_monitor_reports_negative_count():
    agents = [_make_agent("a"), _make_agent("b")]
    _seed_negative(agents[0], [[1.0, 0.0, 0.0, 0.0]])
    _seed_negative(agents[1], [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    store = agents[0].store
    monitor = StructuralLawMonitor(store=store, T_monitor=1, verbose=False)
    result = monitor.step(step_t=1, agents=agents, total_conflict=0.0)

    assert "negative_count" in result
    assert result["negative_count"] == 3  # 1 + 2


# ── Test 2: taboo_overlap computes correctly ──────────────────────────────────

def test_taboo_overlap_no_shared_patterns():
    """Two agents with fully distinct negative UUIDs → overlap = 0.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    _seed_negative(agents[0], [[1.0, 0.0, 0.0, 0.0]])
    _seed_negative(agents[1], [[0.0, 1.0, 0.0, 0.0]])

    overlap = _taboo_overlap(agents)
    # Different UUIDs (GaussianPattern generates fresh UUID each time)
    assert overlap == pytest.approx(0.0)


def test_taboo_overlap_fully_shared():
    """Two agents sharing the same pattern object → overlap = 1.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    p = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.1)
    agents[0].store._tier2_negative.save(p, 0.5, "a")
    agents[1].store._tier2_negative.save(p, 0.5, "b")  # same UUID

    overlap = _taboo_overlap(agents)
    assert overlap == pytest.approx(1.0)


def test_taboo_overlap_empty_stores():
    """No agents have negative patterns → overlap = 0.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    overlap = _taboo_overlap(agents)
    assert overlap == pytest.approx(0.0)


# ── Test 3: taboo_overlap in monitor report ───────────────────────────────────

def test_monitor_reports_taboo_overlap():
    agents = [_make_agent("a"), _make_agent("b")]
    store = agents[0].store
    monitor = StructuralLawMonitor(store=store, T_monitor=1, verbose=False)
    result = monitor.step(step_t=1, agents=agents, total_conflict=0.0)

    assert "taboo_overlap" in result
    assert 0.0 <= result["taboo_overlap"] <= 1.0


# ── Test 4: RecombinationStrategist Fear Reset fires on high taboo_overlap ────

def test_fear_reset_fires_when_taboo_overlap_high():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8

    agents = [_make_agent("a", gamma_neg=0.3), _make_agent("b", gamma_neg=0.3)]
    field_quality = {"taboo_overlap": 0.9, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert result["fear_reset_fired"] is True
    # All agents should have gamma_neg zeroed
    for agent in agents:
        assert agent.config.gamma_neg == pytest.approx(0.0)


# ── Test 5: Fear Reset restores gamma_neg after duration ─────────────────────

def test_fear_reset_restores_gamma_neg():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8
    strategist.fear_reset_duration = 3

    agents = [_make_agent("a", gamma_neg=0.3)]
    high_taboo = {"taboo_overlap": 0.9, "diversity": None, "conflict": 0.0}
    low_taboo = {"taboo_overlap": 0.0, "diversity": None, "conflict": 0.0}

    # Fire reset
    strategist.step(step_t=1, field_quality=high_taboo, agents=agents)
    assert agents[0].config.gamma_neg == pytest.approx(0.0)

    # Tick through duration (3 steps)
    for t in range(2, 5):
        strategist.step(step_t=t, field_quality=low_taboo, agents=agents)

    # After duration, gamma_neg restored to 0.3
    assert agents[0].config.gamma_neg == pytest.approx(0.3)


# ── Test 6: Fear Reset does not fire when taboo_overlap below threshold ───────

def test_fear_reset_does_not_fire_below_threshold():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8

    agents = [_make_agent("a", gamma_neg=0.3)]
    field_quality = {"taboo_overlap": 0.5, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert result["fear_reset_fired"] is False
    assert agents[0].config.gamma_neg == pytest.approx(0.3)


# ── Test 7: Fear Reset result keys present even when not fired ────────────────

def test_fear_reset_result_keys_always_present():
    strategist = RecombinationStrategist()
    agents = [_make_agent("a")]
    field_quality = {"taboo_overlap": 0.0, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert "fear_reset_active" in result
    assert "fear_reset_fired" in result
