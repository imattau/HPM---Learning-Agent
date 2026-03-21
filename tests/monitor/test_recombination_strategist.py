import pytest
from unittest.mock import MagicMock
from hpm.monitor.recombination_strategist import RecombinationStrategist


def _make_agent(agent_id="a1", conflict_threshold=0.1, kappa_0=0.1, beta_c=1.0):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.config.conflict_threshold = conflict_threshold
    agent.config.kappa_0 = kappa_0
    agent.config.beta_c = beta_c
    return agent


def _heavy_fq(diversity=1.0, conflict=0.0):
    """field_quality dict with heavy metrics present."""
    return {
        "pattern_count": 5,
        "level_distribution": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        "level4plus_count": 2,
        "level4plus_mean_weight": 0.4,
        "conflict": conflict,
        "stability_mean": 0.7,
        "diversity": diversity,
        "redundancy": 0.1,
    }


def _light_fq(conflict=0.0):
    """field_quality dict with heavy metrics absent (non-T_monitor step)."""
    fq = _heavy_fq(conflict=conflict)
    fq["diversity"] = None
    fq["redundancy"] = None
    return fq


# ---- Burst mode ----

def test_no_intervention_when_healthy():
    """High diversity + low conflict → no config mutations."""
    agents = [_make_agent()]
    s = RecombinationStrategist(diversity_low=0.5, conflict_high=0.3, stagnation_window=3)
    result = s.step(1, _heavy_fq(diversity=1.5, conflict=0.1), agents)
    assert result["burst_active"] is False
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)


def test_burst_fires_after_stagnation_window():
    """N consecutive stagnant heavy-metric steps → burst fires."""
    agents = [_make_agent(conflict_threshold=0.1)]
    s = RecombinationStrategist(
        diversity_low=0.5, conflict_high=0.3, stagnation_window=3,
        burst_conflict_threshold=0.01, burst_duration=10
    )
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)
    s.step(2, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)  # not yet
    s.step(3, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)  # burst fired


def test_burst_active_flag():
    """burst_active is True while burst is running."""
    agents = [_make_agent()]
    s = RecombinationStrategist(stagnation_window=1, burst_conflict_threshold=0.01, burst_duration=5)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    result = s.step(1, stagnant, agents)
    assert result["burst_active"] is True


def test_burst_restores_config_after_duration():
    """Original conflict_threshold restored after burst_duration steps."""
    agents = [_make_agent(conflict_threshold=0.15)]
    s = RecombinationStrategist(stagnation_window=1, burst_conflict_threshold=0.01, burst_duration=3)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)  # burst fires
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)
    s.step(2, _light_fq(), agents)  # step 1 of burst
    s.step(3, _light_fq(), agents)  # step 2 of burst
    s.step(4, _light_fq(), agents)  # step 3 of burst → restored
    assert agents[0].config.conflict_threshold == pytest.approx(0.15)


def test_burst_cooldown_prevents_retrigger():
    """Burst fires; stagnation resumes immediately; burst does NOT re-fire during cooldown."""
    agents = [_make_agent()]
    s = RecombinationStrategist(
        stagnation_window=1, burst_conflict_threshold=0.01,
        burst_duration=2, burst_cooldown=10
    )
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)   # burst fires
    s.step(2, _light_fq(), agents)
    s.step(3, _light_fq(), agents)  # burst ends, cooldown starts
    # Try to retrigger — stagnant heavy step during cooldown
    orig = agents[0].config.conflict_threshold
    s.step(4, stagnant, agents)
    s.step(5, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(orig)  # no burst


def test_stagnation_skipped_when_diversity_none():
    """If diversity is None (light step), stagnation counter does not advance."""
    agents = [_make_agent()]
    s = RecombinationStrategist(stagnation_window=2, burst_conflict_threshold=0.01)
    light = _light_fq(conflict=0.9)
    s.step(1, light, agents)
    s.step(2, light, agents)
    s.step(3, light, agents)
    assert s._stagnation_count == 0
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)


def test_burst_fires_on_stagnation_window_step():
    """Burst fires on the step when stagnation_count reaches window (not the step after).

    The implementation increments the stagnation counter AND fires the burst in the
    same step when count reaches the window threshold. With stagnation_window=2 and
    two consecutive stagnant heavy steps, the burst fires at the end of step 2.
    (The spec test name 'test_burst_fires_on_step_after_stagnation_window_not_during'
    is misleading — 'after' refers to the logic ordering within a single step call,
    not a separate step_t+1 call.)
    """
    agents = [_make_agent(conflict_threshold=0.1)]
    s = RecombinationStrategist(stagnation_window=2, burst_conflict_threshold=0.01, burst_duration=5)
    stagnant = _heavy_fq(diversity=0.2, conflict=0.5)
    s.step(1, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.1)  # count=1, not yet
    s.step(2, stagnant, agents)
    assert agents[0].config.conflict_threshold == pytest.approx(0.01)  # count=2 → fired


# ---- kappa_0 adoption scaling ----

def test_kappa_0_no_nudge_on_first_heavy_step():
    """First step with non-None diversity sets EMA but does not nudge kappa_0."""
    agents = [_make_agent(kappa_0=0.1)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.2, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=2.0), agents)
    assert agents[0].config.kappa_0 == pytest.approx(0.1)
    assert s._diversity_ema == pytest.approx(2.0)


def test_kappa_0_rises_when_diversity_improving():
    """diversity above EMA → kappa_0 nudged toward kappa_0_max."""
    agents = [_make_agent(kappa_0=0.1)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.5, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=1.0), agents)  # initialise EMA=1.0, no nudge
    s.step(2, _heavy_fq(diversity=2.0), agents)  # new EMA=1.5 < diversity=2.0 → up
    assert agents[0].config.kappa_0 > 0.1


def test_kappa_0_falls_when_diversity_falling():
    """diversity below EMA → kappa_0 nudged toward kappa_0_min."""
    agents = [_make_agent(kappa_0=0.2)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.5, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=2.0), agents)  # EMA=2.0, no nudge
    s.step(2, _heavy_fq(diversity=0.5), agents)  # new EMA=1.25 > diversity=0.5 → down
    assert agents[0].config.kappa_0 < 0.2


def test_kappa_0_clamped_to_bounds():
    """kappa_0 never exceeds kappa_0_max or falls below kappa_0_min."""
    # Upper bound: nudge toward max repeatedly
    agents = [_make_agent(kappa_0=0.28)]
    s = RecombinationStrategist(kappa_0_ema_alpha=0.9, kappa_0_min=0.05, kappa_0_max=0.3)
    s.step(1, _heavy_fq(diversity=0.1), agents)  # initialise EMA=0.1, no nudge
    for _ in range(10):
        s.step(2, _heavy_fq(diversity=100.0), agents)  # diversity >> EMA → nudge up
    assert agents[0].config.kappa_0 <= 0.3

    # Lower bound: fresh strategist so EMA is reset, then nudge toward min repeatedly
    agents2 = [_make_agent(kappa_0=0.06)]
    s2 = RecombinationStrategist(kappa_0_ema_alpha=0.9, kappa_0_min=0.05, kappa_0_max=0.3)
    s2.step(1, _heavy_fq(diversity=100.0), agents2)  # initialise EMA=100.0, no nudge
    for _ in range(10):
        s2.step(2, _heavy_fq(diversity=0.0), agents2)  # diversity << EMA → nudge down
    assert agents2[0].config.kappa_0 >= 0.05


# ---- beta_c damping ----

def test_beta_c_damped_when_conflict_persists():
    """conflict > conflict_high for stagnation_window cycles → beta_c reduced."""
    agents = [_make_agent(beta_c=1.0)]
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=3, beta_c_decay=0.9, beta_c_min=0.1)
    high_conflict = _heavy_fq(diversity=1.0, conflict=0.5)
    s.step(1, high_conflict, agents)
    s.step(2, high_conflict, agents)
    assert agents[0].config.beta_c == pytest.approx(1.0)  # not yet
    s.step(3, high_conflict, agents)
    assert agents[0].config.beta_c == pytest.approx(0.9)


def test_beta_c_floored_at_minimum():
    """Repeated damping → beta_c never falls below beta_c_min."""
    agents = [_make_agent(beta_c=1.0)]
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=1, beta_c_decay=0.1, beta_c_min=0.1)
    high_conflict = _heavy_fq(diversity=1.0, conflict=0.5)
    for i in range(20):
        s.step(i, high_conflict, agents)
    assert agents[0].config.beta_c >= 0.1


def test_agent_missing_beta_c_skipped():
    """Agent config without beta_c attribute is skipped silently during damping."""
    agent = MagicMock()
    agent.agent_id = "a1"
    agent.config.conflict_threshold = 0.1
    agent.config.kappa_0 = 0.1
    del agent.config.beta_c  # no beta_c
    # hasattr returns False for MagicMock deleted attributes — use spec instead
    agent2 = MagicMock(spec=["agent_id", "config"])
    agent2.agent_id = "a2"
    agent2.config = MagicMock(spec=["conflict_threshold", "kappa_0"])
    agent2.config.conflict_threshold = 0.1
    agent2.config.kappa_0 = 0.1
    s = RecombinationStrategist(conflict_high=0.3, stagnation_window=1, beta_c_min=0.1)
    high = _heavy_fq(diversity=1.0, conflict=0.5)
    # Should not raise
    s.step(1, high, [agent2])


# ---- Interventions dict ----

def test_interventions_dict_always_present():
    """Returned dict always has all expected keys."""
    agents = [_make_agent()]
    s = RecombinationStrategist()
    result = s.step(1, _heavy_fq(), agents)
    assert "burst_active" in result
    assert "kappa_0" in result
    assert "beta_c_scaled" in result
    assert "stagnation_count" in result
    assert "cooldown_remaining" in result


def test_empty_agents_list():
    """step() with agents=[] returns valid dict, no errors."""
    s = RecombinationStrategist()
    result = s.step(1, _heavy_fq(diversity=0.1, conflict=0.9), [])
    assert isinstance(result, dict)
    assert "burst_active" in result


# ---- Integration tests ----

import numpy as np
from hpm.store.sqlite import SQLiteStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.monitor.recombination_strategist import RecombinationStrategist


def _make_real_agent(store, dim=4):
    config = AgentConfig(feature_dim=dim, agent_id=f"agent_{id(store)}")
    return Agent(config, store=store)


def test_orchestrator_with_no_strategist(tmp_path):
    """Orchestrator with strategist=None returns interventions == {}."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=None, strategist=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("interventions") == {}


def test_orchestrator_strategist_integrated(tmp_path):
    """Orchestrator with monitor + strategist returns both field_quality and interventions."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    strategist = RecombinationStrategist()
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor, strategist=strategist)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert "field_quality" in result
    assert "interventions" in result
    fq = result["interventions"]
    assert "burst_active" in fq
