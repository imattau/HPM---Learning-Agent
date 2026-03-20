# tests/integration/test_phase3.py
import math
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern
from hpm.metrics.hpm_predictions import social_field_convergence


def test_social_field_convergence_decreasing_returns_negative():
    quality_history = [{"diversity": 1.0 - i * 0.1, "redundancy": 0.0} for i in range(10)]
    slope = social_field_convergence(quality_history)
    assert slope < 0.0


def test_social_field_convergence_flat_returns_near_zero():
    quality_history = [{"diversity": 0.5, "redundancy": 0.0} for _ in range(10)]
    slope = social_field_convergence(quality_history)
    assert abs(slope) < 1e-6


def test_social_field_convergence_increasing_returns_positive():
    quality_history = [{"diversity": i * 0.1, "redundancy": 0.0} for i in range(10)]
    slope = social_field_convergence(quality_history)
    assert slope > 0.0


def test_social_field_convergence_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="at least 2"):
        social_field_convergence([{"diversity": 0.5, "redundancy": 0.0}])


def test_multiagent_shared_seed_registers_all_agents():
    field = PatternField()
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4, gamma_soc=0.5), field=field)
        for i in range(3)
    ]
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    orch.run(np.zeros(4), n_steps=10)
    assert set(field._agent_patterns.keys()) == {"a0", "a1", "a2"}


def test_field_quality_history_computable():
    """Multi-agent run with shared seed: collect field quality, compute convergence slope."""
    field = PatternField()
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4, gamma_soc=0.5, rho=1.0), field=field)
        for i in range(3)
    ]
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    quality_history = []
    for _ in range(30):
        orch.step({f"a{i}": np.zeros(4) for i in range(3)})
        quality_history.append(field.field_quality())

    slope = social_field_convergence(quality_history)
    assert isinstance(slope, float)
    assert math.isfinite(slope)


def test_m3_flag_set_for_single_agent():
    field = PatternField()
    agent = Agent(AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=0.5), field=field)
    orch = MultiAgentOrchestrator([agent], field)
    metrics = orch.step({"solo": np.zeros(4)})
    assert metrics["solo"]["m3_active"] is True


def test_m3_flag_false_for_multi_agent():
    field = PatternField()
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4), field=field)
        for i in range(2)
    ]
    orch = MultiAgentOrchestrator(agents, field)
    metrics = orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    assert metrics["a0"]["m3_active"] is False
